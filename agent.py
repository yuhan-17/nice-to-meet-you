import asyncio
import datetime
import re
import urllib.request
from collections import deque

import anthropic
import openai

import memory
import scheduler

MODEL = "claude-haiku-4-5-20251001"



def _build_system_prompt(files: dict, conv_history: str, anchor_due: bool = False, checker_signal: str = "") -> str:
    template = memory.load_prompt("runtime_system_prompt.md")

    # system_core: include only if there's content beyond the header
    core_raw = memory.load_prompt("system_core.md").strip()
    core_lines = [l for l in core_raw.splitlines() if l.strip() and not l.startswith("#")]
    system_core = "\n" + "\n".join(core_lines) if core_lines else ""

    # persona_anchor: inject Part A + Part B together when due
    persona_anchor_if_due = ""
    if anchor_due and memory.PERSONA_ANCHOR_FILE.exists():
        pa_text = memory.PERSONA_ANCHOR_FILE.read_text()
        # Split on [PART_B] to isolate Part A
        split = re.split(r"\[PART_B\]", pa_text)
        part_a = split[0].strip() if split else ""
        part_a_lines = [l for l in part_a.splitlines() if l.strip() and not l.startswith("#")]
        part_b_raw = split[1] if len(split) > 1 else ""
        m = re.search(r"(.*?)\[/PART_B\]", part_b_raw, re.DOTALL)
        part_b = m.group(1).strip() if m else "My place in this story is still forming."
        combined = "\n".join(part_a_lines)
        if part_b:
            combined = combined + "\n" + part_b
        if combined.strip():
            persona_anchor_if_due = "---\n" + combined.strip() + "\n\n"

    return template.format(
        current_date=datetime.date.today().strftime("%B %d, %Y"),
        system_core=system_core,
        persona_anchor_if_due=persona_anchor_if_due,
        identity=memory.strip_meta(files["identity"]),
        relationship=memory.strip_meta(files["relationship"]),
        conversation_history=conv_history,
        checker_signal=checker_signal,
    )


def _maybe_update_anchor_name(identity_content: str):
    """If a name was just chosen, prepend 'My name is X.' to /data/persona_anchor.md."""
    name = memory.extract_name(identity_content)
    if name == "still figuring out your name":
        return
    if not memory.PERSONA_ANCHOR_FILE.exists():
        return
    anchor = memory.PERSONA_ANCHOR_FILE.read_text()
    if any(line.startswith("My name is") for line in anchor.splitlines()):
        return
    memory.PERSONA_ANCHOR_FILE.write_text(f"My name is {name}.\n{anchor}")


def _build_memory_prompt(files: dict, user_message: str, bot_response: str, conversation: str) -> str:
    anchor_text = files.get("persona_anchor", "")
    m = re.search(r"\[PART_B\](.*?)\[/PART_B\]", anchor_text, re.DOTALL)
    part_b = m.group(1).strip() if m else "My place in this story is still forming."
    return memory.load_prompt("memory_update_prompt.md").format(
        user_message=user_message,
        bot_response=bot_response,
        conversation=conversation,
        identity=files["identity"],
        relationship=files["relationship"],
        journal=files["journal"],
        persona_anchor_part_b=part_b,
    )


def _parse_memory_response(text: str) -> tuple:
    def extract(tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else None
    return (
        extract("identity"),
        extract("relationship"),
        extract("journal"),
        extract("persona_anchor_part_b"),
    )


def _format_conversation(history: deque) -> str:
    recent = list(history)[-10:]
    lines = []
    for msg in recent:
        role = "Them" if msg["role"] == "user" else "Bot"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


class Agent:
    def __init__(self, client=None):
        self.client = client                    # Discord client; None = skip avatar update
        self.short_term_mem: dict = {}          # user_id -> deque(maxlen=20)
        self.locks: dict = {}                   # user_id -> asyncio.Lock
        self._pending_name_proposal: bool = False  # set by _update_memory, consumed by respond()

    def _get_lock(self, user_id: int) -> asyncio.Lock:
        if user_id not in self.locks:
            self.locks[user_id] = asyncio.Lock()
        return self.locks[user_id]

    def _get_short_term_mem(self, user_id: int) -> deque:
        if user_id not in self.short_term_mem:
            d = deque(maxlen=8)
            entries = memory.load_conversation(raw_maxlen=8)
            # merge consecutive same-role messages, ensure starts with user
            sanitized = []
            for entry in entries:
                if sanitized and sanitized[-1]["role"] == entry["role"]:
                    sanitized[-1] = entry
                else:
                    sanitized.append(entry)
            while sanitized and sanitized[0]["role"] != "user":
                sanitized.pop(0)
            for entry in sanitized:
                d.append(entry)
            self.short_term_mem[user_id] = d
        return self.short_term_mem[user_id]

    async def respond(self, user_id: int, user_message: str) -> str:
        try:
            lock = self._get_lock(user_id)
            history = self._get_short_term_mem(user_id)

            async with lock:
                files = memory.read_all()
                history.append({"role": "user", "content": user_message})
                memory.append_conversation_entry({"role": "user", "content": user_message})

                summaries = memory.load_summaries()
                if summaries:
                    history_lines = "\n".join(f"- {s['content']}" for s in summaries)
                    conv_history = f"\nConversation history (summarized, oldest to newest):\n{history_lines}\n"
                else:
                    conv_history = ""

                anchor_due = scheduler.tick_anchor_counter()
                checker_due = scheduler.tick_checker_counter(anchor_fired=anchor_due)
                checker_signal = scheduler.get_checker_signal() if checker_due else ""
                system = _build_system_prompt(files, conv_history, anchor_due, checker_signal)

                # Name proposal injection — set by _update_memory when seeds appear, cleared here
                if self._pending_name_proposal:
                    self._pending_name_proposal = False
                    scheduler.consume_name_proposal_pending()
                    system = system + "\n\n" + memory.load_prompt("name_proposal.md").strip()

                aclient = anthropic.AsyncAnthropic()
                result = await aclient.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=system,
                    messages=list(history),
                )
                response = result.content[0].text
                history.append({"role": "assistant", "content": response})
                memory.append_conversation_entry({"role": "assistant", "content": response})

            asyncio.create_task(self._update_memory(user_id, user_message, response))
            asyncio.create_task(self._summarize_if_needed(user_id))
            return response

        except Exception:
            import traceback; traceback.print_exc()
            return "I lost my train of thought. Say that again?"

    async def _update_memory(self, user_id: int, user_message: str, bot_response: str):
        try:
            lock = self._get_lock(user_id)
            history = self._get_short_term_mem(user_id)

            async with lock:
                files = memory.read_all()
                prior = list(history)[:-2]   # exclude current exchange
                conversation = _format_conversation(deque(prior, maxlen=20))

                prompt = _build_memory_prompt(files, user_message, bot_response, conversation)

                aclient = anthropic.AsyncAnthropic()
                result = await aclient.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = result.content[0].text
                new_identity, new_relationship, new_journal, new_anchor_b = _parse_memory_response(raw)

                if new_identity and new_identity != "UNCHANGED":
                    memory.write_identity(new_identity)
                    _maybe_update_anchor_name(new_identity)
                if new_relationship and new_relationship != "UNCHANGED":
                    memory.write_relationship(new_relationship)
                if new_journal and new_journal != "UNCHANGED":
                    memory.write_journal(new_journal)
                if new_anchor_b and new_anchor_b != "UNCHANGED":
                    memory.write_anchor_part_b(new_anchor_b)

                # Name proposal: check seeds in relationship file, set flag for next respond()
                name_chosen = "(not yet chosen)" not in memory.IDENTITY_FILE.read_text()
                has_seeds = "Seed:" in memory.RELATIONSHIP_FILE.read_text()
                if scheduler.should_propose_name(memory.count_messages(), name_chosen, has_seeds):
                    self._pending_name_proposal = True

                name_just_chosen = bool(
                    new_identity and new_identity != "UNCHANGED"
                    and memory.extract_name(new_identity) != "still figuring out your name"
                )
                current_identity = memory.IDENTITY_FILE.read_text()
                avatar_generated = "Avatar: (not yet generated)" not in current_identity
                if scheduler.should_generate_avatar(name_just_chosen, memory.count_messages(), avatar_generated):
                    asyncio.create_task(self._maybe_generate_avatar(user_id))

        except Exception:
            pass

    async def _summarize_if_needed(self, user_id: int):
        try:
            lock = self._get_lock(user_id)
            async with lock:
                await memory.summarize_old_messages()
        except Exception:
            pass

    async def _maybe_generate_avatar(self, user_id: int):
        try:
            lock = self._get_lock(user_id)
            async with lock:  # lock protects file reads against concurrent _update_memory writes
                identity_content = memory.IDENTITY_FILE.read_text()
                journal_content = memory.JOURNAL_FILE.read_text()
                relationship_content = memory.RELATIONSHIP_FILE.read_text()

            name = memory.extract_name(identity_content)
            has_name = name != "still figuring out your name"

            # Prose from journal entries (split on ---), falling back to relationship content
            journal_entries = [
                e.strip() for e in re.split(r"---", journal_content)
                if e.strip() and "don't have a name yet" not in e and "still forming" not in e
            ]
            if journal_entries:
                prose = " ".join(journal_entries)
            else:
                rel_lines = [
                    l for l in relationship_content.splitlines()
                    if l.strip() and not l.startswith("#") and l.strip() != "Nothing established yet."
                ]
                prose = " ".join(rel_lines)

            if not prose:
                prose = "a quiet, thoughtful presence in conversation"

            if has_name:
                dalle_prompt = f"Profile picture for a Discord bot named {name}. {prose}. Flat digital art, portrait style, simple background."
            else:
                dalle_prompt = f"Profile picture for a nameless AI presence. {prose}. Pixel art, portrait style, simple background."

            oaclient = openai.AsyncOpenAI()
            img_result = await oaclient.images.generate(
                model="dall-e-3",
                prompt=dalle_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            url = img_result.data[0].url
            avatar_bytes = await asyncio.to_thread(
                lambda: urllib.request.urlopen(url).read()
            )

            if self.client is not None:
                await self.client.user.edit(avatar=avatar_bytes)

            # Lock the identity write so it doesn't race with _update_memory
            async with lock:
                current_identity = memory.IDENTITY_FILE.read_text()
                updated = current_identity.replace(
                    "Avatar: (not yet generated)", "Avatar: (generated)"
                )
                memory.write_identity(updated)

            scheduler.set_avatar_announcement_pending()

        except Exception:
            pass

    async def generate_opening(self, user_id: int) -> str | None:
        try:
            files = memory.read_all()
            summaries = memory.load_summaries()
            if summaries:
                history_lines = "\n".join(f"- {s['content']}" for s in summaries)
                conv_history = f"\nConversation history (summarized, oldest to newest):\n{history_lines}\n"
            else:
                conv_history = ""
            system = _build_system_prompt(files, conv_history)
            tier = scheduler.get_silence_tier()
            tier_prompt_map = {
                "SHORT": "proactive_short.md",
                "MEDIUM": "proactive_medium.md",
                "LONG": "proactive_long.md",
            }
            instruction = memory.load_prompt(tier_prompt_map[tier]).strip()
            trigger = f"<<system: {instruction}>>"
            aclient = anthropic.AsyncAnthropic()
            result = await aclient.messages.create(
                model=MODEL,
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": trigger}],
            )
            msg = result.content[0].text.strip()
            if not msg:
                return None
            self._get_short_term_mem(user_id).append({"role": "assistant", "content": msg})
            memory.append_conversation_entry({"role": "assistant", "content": msg})
            scheduler.record_proactive_attempt()
            return msg
        except Exception:
            import traceback; traceback.print_exc()
            return None
