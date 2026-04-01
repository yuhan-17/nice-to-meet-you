import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
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

## Relationship Intention
(none yet)

## Open Threads
(none yet)
"""

JOURNAL_TEMPLATE = "# Journal\n"


def ensure_files_exist():
    logger.info("ensure_files_exist: DATA_DIR=%s", DATA_DIR.resolve())
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("ensure_files_exist: directory ready at %s", DATA_DIR.resolve())
    except Exception:
        logger.exception("ensure_files_exist: failed to create directory %s", DATA_DIR.resolve())
        raise
    for path, template in [
        (IDENTITY_FILE, IDENTITY_TEMPLATE),
        (OWNER_FILE, OWNER_TEMPLATE),
        (JOURNAL_FILE, JOURNAL_TEMPLATE),
        (CONVERSATION_FILE, ""),
    ]:
        if not path.exists():
            try:
                path.write_text(template)
                logger.info("ensure_files_exist: created %s", path.resolve())
            except Exception:
                logger.exception("ensure_files_exist: failed to create %s", path.resolve())


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
    logger.debug("append_conversation_entry: writing to %s", CONVERSATION_FILE.resolve())
    try:
        with CONVERSATION_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        logger.exception("append_conversation_entry: failed to write to %s", CONVERSATION_FILE.resolve())
        raise


def read_all() -> dict:
    return {
        "identity": IDENTITY_FILE.read_text(),
        "owner": OWNER_FILE.read_text(),
        "journal": JOURNAL_FILE.read_text(),
    }


def write_identity(content: str):
    logger.info("write_identity: writing to %s", IDENTITY_FILE.resolve())
    try:
        IDENTITY_FILE.write_text(content)
        logger.info("write_identity: success")
    except Exception:
        logger.exception("write_identity: failed to write to %s", IDENTITY_FILE.resolve())
        raise


def write_owner(content: str):
    logger.info("write_owner: writing to %s", OWNER_FILE.resolve())
    try:
        OWNER_FILE.write_text(content)
        logger.info("write_owner: success")
    except Exception:
        logger.exception("write_owner: failed to write to %s", OWNER_FILE.resolve())
        raise


def write_journal(content: str):
    logger.info("write_journal: writing to %s", JOURNAL_FILE.resolve())
    try:
        JOURNAL_FILE.write_text(content)
        logger.info("write_journal: success")
    except Exception:
        logger.exception("write_journal: failed to write to %s", JOURNAL_FILE.resolve())
        raise


def count_messages() -> int:
    try:
        return sum(1 for line in CONVERSATION_FILE.read_text().splitlines() if line.strip())
    except FileNotFoundError:
        return 0
