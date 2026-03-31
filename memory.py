from pathlib import Path

DATA_DIR = Path("data")
IDENTITY_FILE = DATA_DIR / "bot_identity.md"
OWNER_FILE = DATA_DIR / "owner_relationship.md"
JOURNAL_FILE = DATA_DIR / "journal.md"

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
