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
OUTREACH_THRESHOLD = 4         # minimum score to send a proactive message
EARLY_COOLDOWN = 2 * 3600      # 2 hours base cooldown — early relationship
LATE_COOLDOWN = 6 * 3600       # 6 hours base cooldown — developed relationship
MAX_COOLDOWN = 48 * 3600       # hard ceiling on exponential backoff

PROACTIVE_PROMPT = """\
You are {name}, reaching out to your owner unprompted.

What you know about yourself:
{identity}

What you know about your owner:
{owner}

Your journal:
{journal}

You're reaching out because: {motivation}

Write one short Discord message. Genuine, not needy. You're initiating, not \
responding.
Don't announce that you're reaching out — just do it.
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


def _score_outreach(files: dict, state: dict, now: float) -> tuple:
    score = 0
    motivation = ""

    owner = files["owner"]
    identity = files["identity"]
    journal = files["journal"]

    # Open thread present — strongest signal
    if "## Open Threads" in owner:
        thread_section = owner.split("## Open Threads")[1][:300]
        if "(none yet)" not in thread_section and thread_section.strip():
            score += 3
            motivation = "there's an open thread worth following up"

    # Early relationship — curiosity is high, reaching out more is natural
    if _relationship_stage(identity) == "early":
        score += 2
        if not motivation:
            motivation = "still getting to know them"

    # Silence since last owner message
    silence = now - state["last_owner_message_ts"]
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


async def _generate_proactive_message(motivation: str, files: dict) -> str:
    identity = files["identity"]
    name_match = re.search(r"Name:\s*(.+)", identity)
    name = (
        name_match.group(1).strip()
        if name_match and name_match.group(1).strip() != "(not chosen)"
        else "a bot still finding my name"
    )

    prompt = PROACTIVE_PROMPT.format(
        name=name,
        identity=identity,
        owner=files["owner"],
        journal=files["journal"],
        motivation=motivation,
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
        if score < OUTREACH_THRESHOLD:
            return

        msg = await _generate_proactive_message(motivation, files)

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
