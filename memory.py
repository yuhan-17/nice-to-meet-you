import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
IDENTITY_FILE = DATA_DIR / "bot_identity.md"
RELATIONSHIP_FILE = DATA_DIR / "relationship.md"
JOURNAL_FILE = DATA_DIR / "journal.md"
CONVERSATION_FILE = DATA_DIR / "conversation.jsonl"

IDENTITY_TEMPLATE = """# Identity

Name: (not chosen yet)
Avatar: (not yet generated)

I'm new here. Still figuring out who I am.
"""

RELATIONSHIP_TEMPLATE = """# This person

Name: (not yet known)

Not sure who they are yet.

## Upcoming / Recent Events
(none)

## Next time
(nothing specific)
"""

JOURNAL_TEMPLATE = "# Journal\n"


def ensure_files_exist():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not IDENTITY_FILE.exists():
        IDENTITY_FILE.write_text(IDENTITY_TEMPLATE)
    if not RELATIONSHIP_FILE.exists():
        RELATIONSHIP_FILE.write_text(RELATIONSHIP_TEMPLATE)
    if not JOURNAL_FILE.exists():
        JOURNAL_FILE.write_text(JOURNAL_TEMPLATE)
    if not CONVERSATION_FILE.exists():
        CONVERSATION_FILE.write_text("")


SUMMARIZE_PROMPT = """\
Summarize this conversation excerpt in 3-5 sentences as if briefly telling a mutual friend what happened. Include: what they talked about, the person's mood, anything they mentioned that's worth remembering, anything left unresolved. Be concrete, not analytical. No psychological interpretation.

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
        f"{'Them' if m['role'] == 'user' else 'Bot'}: {m['content']}"
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


def infer_stage(relationship: str) -> str:
    """Derives stage from what the bot actually knows: is the person's name known?"""
    m = re.search(r"Name:\s*(.+)", relationship)
    if not m:
        return "early"
    name = m.group(1).strip()
    return "developing" if name and "(not yet known)" not in name else "early"


def strip_meta(content: str) -> str:
    """Strip markdown headers and implementation-detail lines before injecting into prompts."""
    lines = []
    for line in content.splitlines():
        if line.startswith('#'):
            continue
        if line.startswith('Avatar:'):
            continue
        lines.append(line)
    result = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines))
    return result.strip()


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
    ensure_files_exist()
    return {
        "identity": IDENTITY_FILE.read_text(),
        "relationship": RELATIONSHIP_FILE.read_text(),
        "journal": JOURNAL_FILE.read_text(),
    }


def write_identity(content: str):
    IDENTITY_FILE.write_text(content)


def write_relationship(content: str):
    RELATIONSHIP_FILE.write_text(content)


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
