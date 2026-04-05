import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-6"

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "prompts"))
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))

IDENTITY_FILE = DATA_DIR / "bot_identity.md"
RELATIONSHIP_FILE = DATA_DIR / "relationship_state.md"
JOURNAL_FILE = DATA_DIR / "journal.md"
PERSONA_ANCHOR_FILE = DATA_DIR / "persona_anchor.md"
CONVERSATION_FILE = DATA_DIR / "conversation.jsonl"


def load_prompt(name: str, section: str | None = None) -> str:
    """Read a prompt template from PROMPTS_DIR, optionally extracting a [SECTION] block."""
    text = (PROMPTS_DIR / name).read_text()
    if section is None:
        return text
    m = re.search(rf"\[{section}\](.*?)\[/{section}\]", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _copy_template(src: Path, dest: Path):
    """Copy src to dest if src exists, otherwise write empty string."""
    dest.write_text(src.read_text() if src.exists() else "")


def ensure_files_exist():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Migrate old relationship.md → relationship_state.md
    old_rel = DATA_DIR / "relationship.md"
    if old_rel.exists() and not RELATIONSHIP_FILE.exists():
        RELATIONSHIP_FILE.write_text(old_rel.read_text())
    # Copy doc templates to /data/ on first boot
    if not IDENTITY_FILE.exists():
        _copy_template(DOCS_DIR / "bot_identity.md", IDENTITY_FILE)
    if not RELATIONSHIP_FILE.exists():
        _copy_template(DOCS_DIR / "relationship_state.md", RELATIONSHIP_FILE)
    if not JOURNAL_FILE.exists():
        _copy_template(DOCS_DIR / "journal.md", JOURNAL_FILE)
    if not PERSONA_ANCHOR_FILE.exists():
        _copy_template(PROMPTS_DIR / "persona_anchor.md", PERSONA_ANCHOR_FILE)
    if not CONVERSATION_FILE.exists():
        CONVERSATION_FILE.write_text("")


SUMMARIZE_PROMPT = load_prompt("summarize.md")


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



def load_system_core() -> str:
    """Load system_core.md, stripping headers and blank lines."""
    core_raw = load_prompt("system_core.md").strip()
    core_lines = [l for l in core_raw.splitlines() if l.strip() and not l.startswith("#")]
    return "\n".join(core_lines)


def format_summary_history() -> str:
    """Format conversation summaries into a history block, or empty string if none."""
    summaries = load_summaries()
    if not summaries:
        return ""
    history_lines = "\n".join(f"- {s['content']}" for s in summaries)
    return f"\nConversation history (summarized, oldest to newest):\n{history_lines}\n"


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
    # Treat any parenthetical/uncertain name as not yet chosen
    if name.startswith("(") or "not chosen" in name.lower() or "not yet" in name.lower():
        return "still figuring out your name"
    return name


def append_conversation_entry(entry: dict):
    with CONVERSATION_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def write_anchor_part_b(new_b: str):
    """Rewrite only the content between [PART_B] / [/PART_B] tags in data/persona_anchor.md."""
    if not PERSONA_ANCHOR_FILE.exists():
        return
    text = PERSONA_ANCHOR_FILE.read_text()
    updated = re.sub(
        r"\[PART_B\].*?\[/PART_B\]",
        f"[PART_B]\n{new_b}\n[/PART_B]",
        text,
        flags=re.DOTALL,
    )
    PERSONA_ANCHOR_FILE.write_text(updated)


def read_all() -> dict:
    ensure_files_exist()
    return {
        "identity": IDENTITY_FILE.read_text(),
        "relationship": RELATIONSHIP_FILE.read_text(),
        "journal": JOURNAL_FILE.read_text(),
        "persona_anchor": PERSONA_ANCHOR_FILE.read_text(),
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
