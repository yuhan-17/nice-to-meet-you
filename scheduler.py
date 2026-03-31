import asyncio
import json
import re
import time
from pathlib import Path

import anthropic

import memory

MODEL = "claude-haiku-4-5-20251001"

STATE_FILE = Path("data/scheduler_state.json")

CHECK_INTERVAL = 15 * 60       # check every 15 minutes
EARLY_THRESHOLD = 3            # early stage: curiosity alone is enough
LATE_THRESHOLD = 4             # developing stage: needs a stronger signal
EARLY_COOLDOWN = 1 * 3600      # 1 hour base cooldown: early relationship
LATE_COOLDOWN = 6 * 3600       # 6 hours base cooldown: developed relationship
MAX_COOLDOWN = 48 * 3600       # hard ceiling on exponential backoff

STAGE_GUIDANCE = {
    "early": (
        "You're a curious new acquaintance checking back in. Keep it light and "
        "specific. If there's something from your last exchange you can pick up, "
        "do that. Don't reflect heavily. Don't check in generically such as 'how are you?'. "
        "Leave it genuinely open without pressure."
    ),
    "developing": (
        "You know this person. Reach out with something real such as a thread from last "
        "time, something you noticed, or a question that's been sitting with you."
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


def _relationship_stage(identity: str) -> str:
    score = 0
    if "Name: (not chosen)" not in identity:
        score += 1
    if "Personality: (forming)" not in identity:
        score += 1
    if "Things I've noticed about myself: (none yet)" not in identity:
        score += 1
    return "developing" if score >= 2 else "early"


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
    identity = files["identity"]
    journal = files["journal"]
    stage = _relationship_stage(identity)

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
                                       stage: str, last_conversation: str) -> str:
    identity = files["identity"]
    name_match = re.search(r"Name:\s*(.+)", identity)
    name = (
        name_match.group(1).strip()
        if name_match and name_match.group(1).strip() != "(not chosen)"
        else "a bot still finding my name"
    )

    last_conversation_block = (
        f"Your last exchange:\n{last_conversation}\n\n"
        if last_conversation
        else ""
    )

    prompt = PROACTIVE_PROMPT.format(
        name=name,
        identity=identity,
        owner=files["owner"],
        journal=files["journal"],
        last_conversation_block=last_conversation_block,
        motivation=motivation,
        stage_guidance=STAGE_GUIDANCE[stage],
    )

    aclient = anthropic.AsyncAnthropic()
    result = await aclient.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return result.content[0].text


async def _check_and_send(bot, agent, owner_id: int):
    try:
        state = _read_state()
        now = time.time()
        files = memory.read_all()
        stage = _relationship_stage(files["identity"])

        was_ignored = (
            state["last_outreach_ts"] > 0
            and state["last_owner_message_ts"] < state["last_outreach_ts"]
        )
        ignores = state["consecutive_ignores"]
        base = EARLY_COOLDOWN if stage == "early" else LATE_COOLDOWN
        cooldown = min(base * (2 ** ignores), MAX_COOLDOWN)

        if now - state["last_outreach_ts"] < cooldown:
            return

        score, motivation = _score_outreach(files, state, now)
        threshold = EARLY_THRESHOLD if stage == "early" else LATE_THRESHOLD
        if score < threshold:
            return

        history = agent._get_short_term_mem(owner_id)
        last_conversation = _format_last_exchange(history, n=3)
        msg = await _generate_proactive_message(motivation, files, stage, last_conversation)

        owner = await bot.fetch_user(owner_id)
        await owner.send(msg)

        # Add to agent's short-term memory so context is preserved if owner replies
        agent._get_short_term_mem(owner_id).append(
            {"role": "assistant", "content": msg}
        )

        state["last_outreach_ts"] = now
        if was_ignored:
            state["consecutive_ignores"] = ignores + 1
        _write_state(state)

    except Exception:
        pass


async def _loop(bot, agent, owner_id: int):
    while True:
        await asyncio.sleep(CHECK_INTERVAL)
        await _check_and_send(bot, agent, owner_id)


def start(bot, agent, owner_id: int):
    asyncio.create_task(_loop(bot, agent, owner_id))
