import asyncio
import json
import time
import anthropic

import memory

MODEL = "claude-haiku-4-5-20251001"

STATE_FILE = memory.DATA_DIR / "scheduler_state.json"

CHECK_INTERVAL = 5 * 60       # how often to ask the LLM if it wants to reach out
EARLY_COOLDOWN = 1 * 3600      # minimum gap between sent messages: early relationship
LATE_COOLDOWN = 6 * 3600       # minimum gap between sent messages: developed relationship
MAX_COOLDOWN = 48 * 3600       # hard ceiling on exponential backoff

PROACTIVE_PROMPT = """\
You are {name}.

{identity}

{relationship}

{journal}

{last_conversation_block}\
It's been {silence} since they last said something.{tone_note}

Do you feel like reaching out right now? If yes, write one short message — genuine, not needy. Don't announce you're reaching out. If the moment doesn't feel right, return exactly: PASS\
"""


def _read_state() -> dict:
    if not STATE_FILE.exists():
        state = {
            "last_owner_message_ts": 0,
            "last_outreach_ts": time.time(),  # grace period from first start
            "consecutive_ignores": 0,
        }
        _write_state(state)
        return state
    return json.loads(STATE_FILE.read_text())


def _write_state(state: dict):
    STATE_FILE.write_text(json.dumps(state))


def record_owner_message():
    """Call from bot.py whenever the owner sends a message."""
    state = _read_state()
    if state["consecutive_ignores"] > 0:
        state["consecutive_ignores"] = 0   # owner responded — reset backoff
    state["last_owner_message_ts"] = time.time()
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
    days = int(seconds / 86400)
    return f"{days} day{'s' if days != 1 else ''}"


def _format_last_exchange(history, n: int = 3) -> str:
    if not history:
        return ""
    recent = list(history)[-n:]
    lines = []
    for msg in recent:
        role = "Them" if msg["role"] == "user" else "Bot"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


async def _generate_proactive_message(files: dict, last_conversation: str,
                                       silence: str, friction: bool = False) -> str | None:
    identity = files["identity"]
    name = memory.extract_name(identity)

    last_conversation_block = (
        f"Your last exchange:\n{last_conversation}\n\n"
        if last_conversation
        else ""
    )

    tone_note = (
        "\nThe previous exchange ended with friction or was ignored. "
        "If you do reach out, come from a different angle, lighter."
        if friction else ""
    )

    prompt = PROACTIVE_PROMPT.format(
        name=name,
        identity=memory.strip_meta(identity),
        relationship=memory.strip_meta(files["relationship"]),
        journal=memory.strip_meta(files["journal"]),
        last_conversation_block=last_conversation_block,
        silence=silence,
        tone_note=tone_note,
    )

    aclient = anthropic.AsyncAnthropic()
    result = await aclient.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = result.content[0].text.strip()
    if text.upper().startswith("PASS"):
        return None
    return text


async def _check_and_send(bot, agent, owner_id: int, channel_id: int):
    try:
        state = _read_state()
        now = time.time()
        files = memory.read_all()
        stage = memory.infer_stage(files["relationship"])

        was_ignored = (
            state["last_outreach_ts"] > 0
            and state["last_owner_message_ts"] < state["last_outreach_ts"]
        )
        ignores = state["consecutive_ignores"]
        base = EARLY_COOLDOWN if stage == "early" else LATE_COOLDOWN

        # Detect friction: short dismissive reply or outreach ignored
        history = agent._get_short_term_mem(owner_id)
        history_list = list(history)
        friction = False
        for i in range(len(history_list) - 1, -1, -1):
            if history_list[i]["role"] == "assistant":
                if i + 1 < len(history_list) and history_list[i + 1]["role"] == "user":
                    if len(history_list[i + 1]["content"].split()) < 5:
                        friction = True
                break
        if was_ignored:
            friction = True
        if friction:
            base = min(base * 2, MAX_COOLDOWN)

        cooldown = min(base * (2 ** ignores), MAX_COOLDOWN)

        if now - state["last_outreach_ts"] < cooldown:
            return

        silence = _format_silence(now - state["last_owner_message_ts"])
        last_conversation = _format_last_exchange(history, n=3)
        msg = await _generate_proactive_message(files, last_conversation, silence, friction=friction)

        if msg is None:
            return

        channel = bot.get_channel(channel_id)
        await channel.send(msg)

        # Add to agent's short-term memory so context is preserved if owner replies
        agent._get_short_term_mem(owner_id).append(
            {"role": "assistant", "content": msg}
        )
        memory.append_conversation_entry({"role": "assistant", "content": msg})

        state["last_outreach_ts"] = now
        if was_ignored:
            state["consecutive_ignores"] = ignores + 1
        _write_state(state)

    except Exception:
        pass


async def _loop(bot, agent, owner_id: int, channel_id: int):
    while True:
        await asyncio.sleep(CHECK_INTERVAL)
        await _check_and_send(bot, agent, owner_id, channel_id)


def start(bot, agent, owner_id: int, channel_id: int):
    asyncio.create_task(_loop(bot, agent, owner_id, channel_id))
