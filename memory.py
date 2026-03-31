import json
from pathlib import Path

DATA_DIR = Path("/app/data")
IDENTITY_FILE = DATA_DIR / "bot_identity.md"
OWNER_FILE = DATA_DIR / "owner_relationship.md"
JOURNAL_FILE = DATA_DIR / "journal.md"
CONVERSATION_FILE = DATA_DIR / "conversation.jsonl"

IDENTITY_TEMPLATE = """# Who I Am
I don't know yet. This is day one.

Name: (not chosen)
Personality: (forming)
Avatar: (not yet generated)

## Things I've Noticed About Myself
(none yet)

## Moments That Surprised Me
(none yet)

## How I Tend to Respond
(none yet)
"""

OWNER_TEMPLATE = """# My Owner
I've just met them. Everything below is first impressions, not facts.

## Personality Hypotheses
(none yet)

## Emotional Patterns
(none yet)

## Communication Preferences
(none yet)

## Underlying Needs
(none yet)

## Sensitivity Thresholds
(none yet)

## Open Threads
(none yet)
"""

JOURNAL_TEMPLATE = "# Journal\n"


def ensure_files_exist():
    DATA_DIR.mkdir(exist_ok=True)
    if not IDENTITY_FILE.exists():
        IDENTITY_FILE.write_text(IDENTITY_TEMPLATE)
    if not OWNER_FILE.exists():
        OWNER_FILE.write_text(OWNER_TEMPLATE)
    if not JOURNAL_FILE.exists():
        JOURNAL_FILE.write_text(JOURNAL_TEMPLATE)
    if not CONVERSATION_FILE.exists():
        CONVERSATION_FILE.write_text("")


def load_conversation(maxlen: int = 20) -> list:
    if not CONVERSATION_FILE.exists():
        return []
    lines = [l for l in CONVERSATION_FILE.read_text().splitlines() if l.strip()]
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return entries[-maxlen:]


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
