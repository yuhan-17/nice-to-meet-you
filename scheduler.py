import asyncio
import json
import time
import anthropic

import memory

MODEL = "claude-haiku-4-5-20251001"

STATE_FILE = memory.DATA_DIR / "scheduler_state.json"

CHECK_INTERVAL = 5 * 60       # how often _check_and_send fires

TIER_BACKOFFS = {
    "SHORT":  5 * 60,
    "MEDIUM": 30 * 60,
    "LONG":   4 * 3600,
}

ANCHOR_INJECTION_INTERVAL = 8   # re-inject persona anchor every N conversation turns
CHECKER_INJECTION_INTERVAL = 10  # inject checker signal every N conversation turns


def _read_state() -> dict:
    if not STATE_FILE.exists():
        state = {
            "last_owner_message_ts": 0,
            "turns_since_anchor_injection": 0,
            "turns_since_checker_injection": 0,
            "avatar_prompt_fired": False,
            "name_proactive_fired": False,
            "avatar_announcement_pending": False,
            "last_proactive_attempt_ts": 0,
        }
        _write_state(state)
        return state
    return json.loads(STATE_FILE.read_text())


def get_checker_signal() -> str:
    """Returns a silence note for the system prompt if the user has been quiet for a while."""
    state = _read_state()
    last_msg = state.get("last_owner_message_ts", 0)
    if last_msg == 0:
        return ""
    silence = time.time() - last_msg
    if silence < 1800:  # less than 30 minutes — not worth noting
        return ""
    return f"\n<<system: {_format_silence(silence)} since their last message>>"


def tick_anchor_counter() -> bool:
    """Increment the per-turn counter. Returns True (and resets) when injection is due."""
    state = _read_state()
    count = state.get("turns_since_anchor_injection", 0) + 1
    if count >= ANCHOR_INJECTION_INTERVAL:
        state["turns_since_anchor_injection"] = 0
        _write_state(state)
        return True
    state["turns_since_anchor_injection"] = count
    _write_state(state)
    return False


def tick_checker_counter(anchor_fired: bool = False) -> bool:
    """Increment the checker counter. Returns True (and resets) when injection is due.
    If anchor fired this same turn, suppresses checker and still resets the counter."""
    state = _read_state()
    count = state.get("turns_since_checker_injection", 0) + 1
    if count >= CHECKER_INJECTION_INTERVAL:
        state["turns_since_checker_injection"] = 0
        _write_state(state)
        return False if anchor_fired else True
    state["turns_since_checker_injection"] = count
    _write_state(state)
    return False


def get_silence_tier() -> str:
    """Returns SHORT / MEDIUM / LONG based on time since last owner message."""
    state = _read_state()
    last_msg = state.get("last_owner_message_ts", 0)
    if last_msg == 0:
        return "LONG"
    silence = time.time() - last_msg
    if silence < 600:
        return "SHORT"
    if silence < 10800:
        return "MEDIUM"
    return "LONG"


def consume_name_proposal_pending():
    """Clear the in-memory flag without writing state — seeds path is disabled."""
    pass


NAME_PROACTIVE_THRESHOLD = 20  # raw messages before proactive name raise fires


def should_proactively_propose_name(message_count: int, name_chosen: bool) -> bool:
    """Returns True once, after NAME_PROACTIVE_THRESHOLD messages, if name still not chosen."""
    if name_chosen:
        return False
    state = _read_state()
    if state.get("name_proactive_fired", False):
        return False
    if message_count >= NAME_PROACTIVE_THRESHOLD:
        state["name_proactive_fired"] = True
        _write_state(state)
        return True
    return False



def should_generate_avatar(name_just_chosen: bool, exchange_count: int, avatar_generated: bool) -> bool:
    """Returns True once when avatar generation should fire. Never fires again after that."""
    if avatar_generated:
        return False
    state = _read_state()
    if state.get("avatar_prompt_fired", False):
        return False
    if name_just_chosen or exchange_count >= 10:
        state["avatar_prompt_fired"] = True
        _write_state(state)
        return True
    return False


def set_avatar_announcement_pending():
    state = _read_state()
    state["avatar_announcement_pending"] = True
    _write_state(state)


def reset_avatar_prompt_fired():
    """Clear the one-time flag so generation can retry after a failure."""
    state = _read_state()
    state["avatar_prompt_fired"] = False
    _write_state(state)


def _build_proactive_system() -> str:
    """Assemble system_core + identity + relationship for proactive LLM calls."""
    core_raw = memory.load_prompt("system_core.md").strip()
    core_lines = [l for l in core_raw.splitlines() if l.strip() and not l.startswith("#")]
    files = memory.read_all()
    parts = ["\n".join(core_lines), memory.strip_meta(files["identity"]), memory.strip_meta(files["relationship"])]
    return "\n\n---\n\n".join(p for p in parts if p.strip())


async def _passes_proactive_gate(msg: str) -> bool:
    """Returns True if the message is directed outward, False if it centers the bot's internal state."""
    gate_prompt = memory.load_prompt("proactive_gate.md").format(message=msg)
    aclient = anthropic.AsyncAnthropic()
    result = await aclient.messages.create(
        model=MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": gate_prompt}],
    )
    verdict = result.content[0].text.strip().upper()
    return verdict.startswith("SEND")


def _write_state(state: dict):
    STATE_FILE.write_text(json.dumps(state))


def record_owner_message():
    """Call from bot.py whenever the owner sends a message."""
    state = _read_state()
    state["last_owner_message_ts"] = time.time()
    _write_state(state)


def record_proactive_attempt():
    """Call after generate_opening() sends successfully."""
    state = _read_state()
    state["last_proactive_attempt_ts"] = time.time()
    _write_state(state)



def _format_silence(seconds: float) -> str:
    if seconds < 60:
        return "less than a minute"
    if seconds < 3600:
        mins = int(seconds / 60)
        return f"{mins} minute{'s' if mins != 1 else ''}"
    if seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''}"
    if seconds > 30 * 86400:
        return "a while"
    days = int(seconds / 86400)
    return f"{days} day{'s' if days != 1 else ''}"



async def _check_and_send(bot, agent, owner_id: int, channel_id: int):
    try:
        state = _read_state()

        # Avatar announcement — priority path
        if state.get("avatar_announcement_pending", False):
            state["avatar_announcement_pending"] = False
            _write_state(state)
            core_raw = memory.load_prompt("system_core.md").strip()
            core_lines = [l for l in core_raw.splitlines() if l.strip() and not l.startswith("#")]
            system_core = "\n".join(core_lines)
            announcement_instruction = memory.load_prompt("avatar_announcement.md").strip()
            aclient = anthropic.AsyncAnthropic()
            ann_result = await aclient.messages.create(
                model=MODEL,
                max_tokens=128,
                system=system_core,
                messages=[{"role": "user", "content": f"<<system: {announcement_instruction}>>"}],
            )
            ann_msg = ann_result.content[0].text.strip()
            if ann_msg:
                channel = bot.get_channel(channel_id)
                await channel.send(ann_msg)
                agent._get_short_term_mem(owner_id).append({"role": "assistant", "content": ann_msg})
                memory.append_conversation_entry({"role": "assistant", "content": ann_msg})
            return

        # No proactive fires into an active conversation
        now = time.time()
        last_msg = state.get("last_owner_message_ts", 0)
        if last_msg and now - last_msg < 120:
            return

        # Proactive outreach — single timestamp, per-tier minimum
        tier = get_silence_tier()
        backoff = TIER_BACKOFFS[tier]
        last_attempt = state.get("last_proactive_attempt_ts", 0)
        if now - last_attempt < backoff:
            return

        instruction = memory.load_prompt(f"proactive_{tier.lower()}.md").strip()
        system = _build_proactive_system()
        aclient = anthropic.AsyncAnthropic()
        result = await aclient.messages.create(
            model=MODEL,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": f"<<system: {instruction}>>"}],
        )
        msg = result.content[0].text.strip()

        state["last_proactive_attempt_ts"] = now
        _write_state(state)

        if not msg or not await _passes_proactive_gate(msg):
            return

        channel = bot.get_channel(channel_id)
        await channel.send(msg)
        agent._get_short_term_mem(owner_id).append({"role": "assistant", "content": msg})
        memory.append_conversation_entry({"role": "assistant", "content": msg})

    except Exception:
        pass


async def _loop(bot, agent, owner_id: int, channel_id: int):
    while True:
        await asyncio.sleep(CHECK_INTERVAL)
        await _check_and_send(bot, agent, owner_id, channel_id)


def start(bot, agent, owner_id: int, channel_id: int):
    asyncio.create_task(_loop(bot, agent, owner_id, channel_id))
