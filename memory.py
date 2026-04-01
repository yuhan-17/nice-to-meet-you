import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
IDENTITY_FILE = DATA_DIR / "bot_identity.md"
OWNER_FILE = DATA_DIR / "owner_relationship.md"
JOURNAL_FILE = DATA_DIR / "journal.md"
CONVERSATION_FILE = DATA_DIR / "conversation.jsonl"

IDENTITY_TEMPLATE = """# Who I Am

Name: (not chosen yet)

## Character
(forming)

## What I Actually Think
(forming)

## How I Tend to Be
(forming)
"""

OWNER_TEMPLATE = """# My Owner

Owner's name: (not yet known)

## Who They Are
(not yet known)

## What They're Dealing With Lately
(not yet known)

## How They Like to Talk
(not yet known)

## Current Mode
casual

## Things Worth Remembering
(none yet)

## Upcoming / Recent Events
(none yet)

## Open Threads
(none yet)
"""

JOURNAL_TEMPLATE = "# Journal\n"


def ensure_files_exist():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not IDENTITY_FILE.exists():
        IDENTITY_FILE.write_text(IDENTITY_TEMPLATE)
    if not OWNER_FILE.exists():
        OWNER_FILE.write_text(OWNER_TEMPLATE)
    if not JOURNAL_FILE.exists():
        JOURNAL_FILE.write_text(JOURNAL_TEMPLATE)
    if not CONVERSATION_FILE.exists():
        CONVERSATION_FILE.write_text("")


SUMMARIZE_PROMPT = """\
Summarize this conversation excerpt in 3-5 sentences as if briefly telling a mutual friend what happened. Include: what they talked about, the owner's mood, anything the owner mentioned that's worth remembering, anything left unresolved. Be concrete, not analytical. No psychological interpretation.

{messages}
"""


def load_conversation(raw_maxlen: int = 8) -> list:
    """Returns only raw (non-summary) entries, up to raw_maxlen most recent."""
    if not CONVERSATION_FILE.exists():
        return []
    lines = [l for l in CONVERSATION_FILE.read_text().splitlines() if l.strip()]
    raw = []
    for line in lines:
        try:
            entry = json.loads(line)
            if entry.get("role") != "summary":
                raw.append(entry)
        except Exception:
            pass
    return raw[-raw_maxlen:]


def load_summaries(days: int = 30) -> list:
    """Returns summary entries from the last N days, oldest to newest."""
    if not CONVERSATION_FILE.exists():
        return []
    lines = [l for l in CONVERSATION_FILE.read_text().splitlines() if l.strip()]
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    summaries = []
    for line in lines:
        try:
            entry = json.loads(line)
            if entry.get("role") == "summary":
                try:
                    ts = datetime.fromisoformat(entry.get("timestamp", ""))
                    if ts >= cutoff:
                        summaries.append(entry)
                except Exception:
                    summaries.append(entry)
        except Exception:
            pass
    return summaries


async def summarize_old_messages() -> bool:
    """If raw message count exceeds 40, summarize the oldest 20 and replace them."""
    if not CONVERSATION_FILE.exists():
        return False

    lines = [l for l in CONVERSATION_FILE.read_text().splitlines() if l.strip()]
    all_entries = []
    for line in lines:
        try:
            all_entries.append(json.loads(line))
        except Exception:
            pass

    raw_indices = [i for i, e in enumerate(all_entries) if e.get("role") != "summary"]
    if len(raw_indices) <= 40:
        return False

    oldest_indices = set(raw_indices[:20])
    oldest_msgs = [all_entries[i] for i in raw_indices[:20]]

    formatted = "\n".join(
        f"{'Owner' if m['role'] == 'user' else 'Bot'}: {m['content']}"
        for m in oldest_msgs
    )
    prompt = SUMMARIZE_PROMPT.format(messages=formatted)

    aclient = anthropic.AsyncAnthropic()
    result = await aclient.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    summary_text = result.content[0].text.strip()

    summary_entry = {
        "role": "summary",
        "content": summary_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    new_entries = []
    inserted = False
    for i, entry in enumerate(all_entries):
        if i in oldest_indices:
            if not inserted:
                new_entries.append(summary_entry)
                inserted = True
        else:
            new_entries.append(entry)

    CONVERSATION_FILE.write_text("".join(json.dumps(e) + "\n" for e in new_entries))
    return True


def infer_stage(owner: str) -> str:
    sections = re.split(r"^## .+", owner, flags=re.MULTILINE)
    filled = sum(
        1 for s in sections[1:]
        if s.strip() and "(none yet)" not in s and "(not yet known)" not in s
    )
    return "developing" if filled >= 2 else "early"


def extract_name(identity: str) -> str:
    m = re.search(r"Name:\s*(.+)", identity)
    if not m:
        return "still figuring out your name"
    name = m.group(1).strip()
    if "(not chosen" in name:
        return "still figuring out your name"
    return name


def append_conversation_entry(entry: dict):
    with CONVERSATION_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def read_all() -> dict:
    return {
        "identity": IDENTITY_FILE.read_text(),
        "owner": OWNER_FILE.read_text(),
        "journal": JOURNAL_FILE.read_text(),
    }


def write_identity(content: str):
    IDENTITY_FILE.write_text(content)


def write_owner(content: str):
    OWNER_FILE.write_text(content)


def write_journal(content: str):
    JOURNAL_FILE.write_text(content)


def count_messages() -> int:
    try:
        return sum(
            1 for line in CONVERSATION_FILE.read_text().splitlines()
            if line.strip() and json.loads(line).get("role") != "summary"
        )
    except Exception:
        return 0
