import asyncio
import json
import time
import anthropic

import memory

MODEL = memory.MODEL

STATE_FILE = memory.DATA_DIR / "scheduler_state.json"

CHECK_INTERVAL = 10 * 60         # how often the loop fires

ANCHOR_INJECTION_INTERVAL = 8   # re-inject persona anchor every N conversation turns


PROACTIVE_BASE_INTERVAL = 30 * 60    # 30 min base between proactive messages
PROACTIVE_MAX_INTERVAL = 4 * 3600    # cap at 4 hours


def _read_state() -> dict:
    if not STATE_FILE.exists():
        state = {
            "last_owner_message_ts": 0,
            "last_silence_gap": 0,
            "turns_since_anchor_injection": 0,
            "avatar_prompt_fired": False,
            "name_proactive_fired": False,
            "avatar_announcement_pending": False,
            "last_proactive_attempt_ts": 0,
            "consecutive_unanswered_proactive": 0,
        }
        _write_state(state)
        return state
    return json.loads(STATE_FILE.read_text())


def _write_state(state: dict):
    STATE_FILE.write_text(json.dumps(state))


# --- Anchor injection + return-from-silence context ---

def get_return_context() -> str:
    """If the user just returned after 30+ min of silence, note it."""
    state = _read_state()
    gap = state.get("last_silence_gap", 0)
    if gap < 1800:
        return ""
    return f"\n<<system: It's been {_format_silence(gap)} since they last messaged — they just came back.>>"


def tick_anchor_counter() -> bool:
    state = _read_state()
    count = state.get("turns_since_anchor_injection", 0) + 1
    if count >= ANCHOR_INJECTION_INTERVAL:
        state["turns_since_anchor_injection"] = 0
        _write_state(state)
        return True
    state["turns_since_anchor_injection"] = count
    _write_state(state)
    return False


# --- Name / avatar one-shot triggers ---

NAME_PROACTIVE_THRESHOLD = 20


def should_proactively_propose_name(message_count: int, name_chosen: bool) -> bool:
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
    state = _read_state()
    state["avatar_prompt_fired"] = False
    _write_state(state)


# --- Timestamps ---

def record_owner_message():
    state = _read_state()
    now = time.time()
    prev = state.get("last_owner_message_ts", 0)
    state["last_silence_gap"] = now - prev if prev else 0
    state["last_owner_message_ts"] = now
    state["consecutive_unanswered_proactive"] = 0
    _write_state(state)


# --- Proactive outreach ---

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


def _build_proactive_system() -> str:
    files = memory.read_all()
    parts = [memory.load_system_core(), memory.strip_meta(files["identity"]), memory.strip_meta(files["relationship"]), memory.strip_meta(files["journal"])]

    conv_history = memory.format_summary_history().strip()
    if conv_history:
        parts.append(conv_history)

    return "\n\n---\n\n".join(p for p in parts if p.strip())


def _get_silence_description() -> str:
    state = _read_state()
    last_msg = state.get("last_owner_message_ts", 0)
    if last_msg == 0:
        return "a while"
    return _format_silence(time.time() - last_msg)


async def _check_and_send(bot, agent, owner_id: int, channel_id: int):
    try:
        state = _read_state()

        # Avatar announcement — priority path
        if state.get("avatar_announcement_pending", False):
            state["avatar_announcement_pending"] = False
            _write_state(state)
            system_core = memory.load_system_core()
            announcement_instruction = memory.load_prompt("avatar_announcement.md", "ANNOUNCEMENT")
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

        # Don't fire into an active conversation
        now = time.time()
        last_msg = state.get("last_owner_message_ts", 0)
        if last_msg and now - last_msg < 300:
            return

        # Dynamic interval: backs off exponentially when ignored
        unanswered = state.get("consecutive_unanswered_proactive", 0)
        backoff = min(2 ** unanswered, 8)
        min_interval = min(PROACTIVE_BASE_INTERVAL * backoff, PROACTIVE_MAX_INTERVAL)

        last_attempt = state.get("last_proactive_attempt_ts", 0)
        if now - last_attempt < min_interval:
            return

        # One prompt — LLM decides whether to speak or stay silent
        silence = _get_silence_description()
        instruction = memory.load_prompt("proactive.md").format(silence=silence)
        system = _build_proactive_system()

        aclient = anthropic.AsyncAnthropic()
        result = await aclient.messages.create(
            model=MODEL,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": f"<<system: {instruction}>>"}],
        )
        msg = result.content[0].text.strip()

        state = _read_state()
        state["last_proactive_attempt_ts"] = now

        # "." means the LLM chose silence
        if not msg or msg == ".":
            _write_state(state)
            return

        state["consecutive_unanswered_proactive"] = unanswered + 1
        _write_state(state)

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
