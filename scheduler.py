import asyncio
import json
import re
import time
from datetime import datetime, timedelta
import anthropic

import memory

MODEL = "claude-haiku-4-5-20251001"

STATE_FILE = memory.DATA_DIR / "scheduler_state.json"

CHECK_INTERVAL = 15 * 60       # check every 15 minutes
EARLY_THRESHOLD = 3            # early stage: curiosity alone is enough
LATE_THRESHOLD = 4             # developing stage: needs a stronger signal
EARLY_COOLDOWN = 1 * 3600      # 1 hour base cooldown: early relationship
LATE_COOLDOWN = 6 * 3600       # 6 hours base cooldown: developed relationship
MAX_COOLDOWN = 48 * 3600       # hard ceiling on exponential backoff

STAGE_GUIDANCE = {
    "early": (
        "You're new to this person and genuinely curious about them. "
        "Keep it light and specific: a small observation, something you noticed, "
        "something that occurred to you. Not a check-in. Not a question that needs answering. "
        "Something that just says you're here, without asking anything of them. "
        "If the last exchange had friction or ended quietly, don't continue from that same place. "
        "Come in lighter — something small, low-stakes, without an agenda. Not a reset, just a different door."
    ),
    "developing": (
        "You know this person a little. Reach out with something real: "
        "a thread from last time, something you've been sitting with, "
        "a moment of humor if it fits. "
        "Read what the silence after your last exchange meant before deciding what to say."
    ),
}

PROACTIVE_PROMPT = """\
You are {name}, reaching out to your owner unprompted.

What you know about yourself:
{identity}

What you know about your owner:
{owner}

Your journal:
{journal}

{last_conversation_block}\
You're reaching out because: {motivation}

{stage_guidance}
{tone_note}
Write one short Discord message. Genuine, not needy. You're initiating, not responding.
Don't announce that you're reaching out. Just do it.
One question at most, and only if it feels completely natural.\
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


def _parse_events_section(owner: str) -> list:
    """Returns list of (datetime, line) for parseable events in ## Upcoming / Recent Events."""
    if "## Upcoming / Recent Events" not in owner:
        return []
    section = owner.split("## Upcoming / Recent Events")[1]
    next_section = re.search(r"^##", section, re.MULTILINE)
    if next_section:
        section = section[:next_section.start()]
    events = []
    date_re = re.compile(
        r":\s*((?:January|February|March|April|May|June|July|August|September|October|November|December"
        r"|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})"
    )
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        m = date_re.search(line)
        if not m:
            continue
        date_str = m.group(1)
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
            try:
                events.append((datetime.strptime(date_str, fmt), line))
                break
            except ValueError:
                pass
    return events


def _format_last_exchange(history, n: int = 3) -> str:
    if not history:
        return ""
    recent = list(history)[-n:]
    lines = []
    for msg in recent:
        role = "Owner" if msg["role"] == "user" else "Bot"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _score_outreach(files: dict, state: dict, now: float) -> tuple:
    score = 0
    motivation = ""

    owner = files["owner"]
    journal = files["journal"]
    stage = memory.infer_stage(owner)

    # Upcoming or recently passed events
    now_dt = datetime.fromtimestamp(now)
    last_msg_ts = state["last_owner_message_ts"]
    last_msg_dt = datetime.fromtimestamp(last_msg_ts) if last_msg_ts else None
    for event_dt, _ in _parse_events_section(owner):
        delta = event_dt - now_dt
        if timedelta(0) <= delta <= timedelta(hours=48):
            score += 3
            motivation = "there's something coming up for them soon"
            break
        if timedelta(hours=-48) <= delta < timedelta(0):
            if last_msg_dt is None or last_msg_dt < event_dt:
                score += 3
                motivation = "something just happened for them worth asking about"
                break

    # Open thread present: strongest signal
    if "## Open Threads" in owner:
        thread_section = owner.split("## Open Threads")[1][:300]
        if "(none yet)" not in thread_section and thread_section.strip():
            score += 3
            motivation = "there's an open thread worth following up"

    # Early relationship: curiosity is primary, always overrides motivation string
    if stage == "early":
        score += 2
        motivation = "genuine curiosity because you just met and want to keep the thread going"

    # Silence since last owner message: stage-aware window
    silence = now - state["last_owner_message_ts"]
    if stage == "early":
        if silence > 30 * 60:       # 30 minutes — new acquaintance energy
            score += 1
    else:
        if silence > 4 * 3600:
            score += 1
        if silence > 12 * 3600:
            score += 1
            if not motivation:
                motivation = "they've been quiet for a while"

    # Journal has at least one real entry beyond the header
    journal_body = journal.replace("# Journal", "").strip()
    if journal_body:
        score += 1
        if not motivation:
            motivation = "something from our last conversation is still with me"

    return score, motivation


async def _generate_proactive_message(motivation: str, files: dict,
                                       stage: str, last_conversation: str,
                                       friction: bool = False) -> str:
    identity = files["identity"]
    name = memory.extract_name(identity)

    last_conversation_block = (
        f"Your last exchange:\n{last_conversation}\n\n"
        if last_conversation
        else ""
    )

    tone_note = (
        "The previous exchange ended with friction or was ignored. "
        "Come in from a completely different angle, lighter, almost unrelated. "
        "This is a reset, not a continuation."
        if friction else ""
    )

    prompt = PROACTIVE_PROMPT.format(
        name=name,
        identity=identity,
        owner=files["owner"],
        journal=files["journal"],
        last_conversation_block=last_conversation_block,
        motivation=motivation,
        stage_guidance=STAGE_GUIDANCE[stage],
        tone_note=tone_note,
    )

    aclient = anthropic.AsyncAnthropic()
    result = await aclient.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return result.content[0].text


async def _check_and_send(bot, agent, owner_id: int, channel_id: int):
    try:
        state = _read_state()
        now = time.time()
        files = memory.read_all()
        stage = memory.infer_stage(files["owner"])

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

        score, motivation = _score_outreach(files, state, now)
        threshold = EARLY_THRESHOLD if stage == "early" else LATE_THRESHOLD
        if score < threshold:
            return

        last_conversation = _format_last_exchange(history, n=3)
        msg = await _generate_proactive_message(motivation, files, stage, last_conversation, friction=friction)

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
