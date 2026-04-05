import asyncio
import datetime
import re
import traceback
import urllib.request
from collections import deque

import anthropic
import openai

import memory
import scheduler

MODEL = memory.MODEL

TOOLS = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]



def _build_system_prompt(files: dict, conv_history: str, anchor_due: bool = False, context_signal: str = "") -> str:
    template = memory.load_prompt("runtime_system_prompt.md")

    core = memory.load_system_core()
    system_core = "\n" + core if core else ""

    # persona_anchor: inject Part A + Part B together when due
    persona_anchor_if_due = ""
    if anchor_due and memory.PERSONA_ANCHOR_FILE.exists():
        pa_text = memory.PERSONA_ANCHOR_FILE.read_text()
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
        context_signal=context_signal,
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
        # history entries are always plain text
        content = msg["content"] if isinstance(msg["content"], str) else "[image]"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _collect_text(content_blocks: list) -> str:
    """Concatenate all text blocks from an API response (web search may add multiple)."""
    parts = []
    for block in content_blocks:
        if hasattr(block, "type") and block.type == "text":
            parts.append(block.text)
    return "".join(parts)


class Agent:
    def __init__(self, client=None):
        self.client = client                    # Discord client; None = skip avatar update
        self.short_term_mem: dict = {}          # user_id -> deque(maxlen=20)
        self.locks: dict = {}                   # user_id -> asyncio.Lock
        self._avatar_generation_in_flight: bool = False  # prevents duplicate concurrent generations
        self._last_nick: str | None = None               # last nickname pushed to Discord

    def _get_lock(self, user_id: int) -> asyncio.Lock:
        if user_id not in self.locks:
            self.locks[user_id] = asyncio.Lock()
        return self.locks[user_id]

    def _get_short_term_mem(self, user_id: int) -> deque:
        if user_id not in self.short_term_mem:
            d = deque(maxlen=8)
            entries = memory.load_conversation(raw_maxlen=8)
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

    async def respond(
        self,
        user_id: int,
        user_message: str,
        image_urls: list[str] | None = None,
    ) -> tuple[str, bytes | None]:
        """Returns (text_response, image_bytes_or_None)."""
        try:
            lock = self._get_lock(user_id)
            history = self._get_short_term_mem(user_id)

            async with lock:
                files = memory.read_all()

                # Build user content — multimodal if images attached
                if image_urls:
                    user_content = []
                    if user_message:
                        user_content.append({"type": "text", "text": user_message})
                    for url in image_urls:
                        user_content.append({"type": "image", "source": {"type": "url", "url": url}})
                else:
                    user_content = user_message

                # History stores plain text only; multimodal content is passed inline per-call
                history.append({"role": "user", "content": user_message})
                memory.append_conversation_entry({"role": "user", "content": user_message})

                conv_history = memory.format_summary_history()

                anchor_due = scheduler.tick_anchor_counter()
                return_context = scheduler.get_return_context()
                system = _build_system_prompt(files, conv_history, anchor_due, return_context)

                # Name injection — reactive and proactive paths
                name_chosen = memory.extract_name(files["identity"]) != "still figuring out your name"
                if not name_chosen:
                    msg_lower = user_message.lower()
                    name_relevant = any(p in msg_lower for p in [
                        "what's your name", "what is your name", "your name",
                        "do you have a name", "pick a name", "choose a name",
                        "call yourself", "what should i call you", "who are you",
                        "my name is ", "call me ", "name's ",
                    ])
                    proactive = not name_relevant and scheduler.should_proactively_propose_name(memory.count_messages(), name_chosen)
                    if name_relevant:
                        system = system + "\n\n" + memory.load_prompt("name_proposal.md", "REACTIVE")
                    elif proactive:
                        system = system + "\n\n" + memory.load_prompt("name_proposal.md", "PROACTIVE")

                # Avatar signal — inject instruction when not yet generated so LLM can self-trigger
                avatar_generated = "Avatar: (not yet generated)" not in files["identity"]
                if not avatar_generated:
                    system = system + "\n\n" + memory.load_prompt("avatar_announcement.md", "SIGNAL")

                # Build messages — replace last user entry with multimodal content if needed
                api_messages = list(history)[:-1] + [{"role": "user", "content": user_content}]

                aclient = anthropic.AsyncAnthropic()
                result = await aclient.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=system,
                    tools=TOOLS,
                    messages=api_messages,
                )
                response = _collect_text(result.content)

                # Avatar generation signal
                if not avatar_generated and re.search(r"<generate_avatar\s*/>", response):
                    response = re.sub(r"<generate_avatar\s*/>", "", response).strip()
                    self._trigger_avatar_generation(user_id)

                # Image generation signal
                image_bytes = None
                img_match = re.search(r'<generate_image\s+prompt="([^"]+)"\s*/>', response)
                if img_match:
                    dalle_prompt = img_match.group(1)
                    response = re.sub(r'<generate_image\s+prompt="[^"]+"\s*/>', "", response).strip()
                    image_bytes = await self._generate_image(dalle_prompt)

                history.append({"role": "assistant", "content": response})
                memory.append_conversation_entry({"role": "assistant", "content": response})

            asyncio.create_task(self._update_memory(user_id, user_message, response))
            asyncio.create_task(self._summarize_if_needed(user_id))
            return response, image_bytes

        except Exception:
            traceback.print_exc()
            return "I lost my train of thought. Say that again?", None

    async def _generate_image(self, prompt: str) -> bytes | None:
        try:
            oaclient = openai.AsyncOpenAI()
            result = await oaclient.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            url = result.data[0].url
            return await asyncio.to_thread(lambda: urllib.request.urlopen(url).read())
        except Exception:
            traceback.print_exc()
            return None

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
                    name = memory.extract_name(new_identity)
                    _maybe_update_anchor_name(new_identity)
                    await self._update_discord_nickname(name)
                else:
                    name = None
                if new_relationship and new_relationship != "UNCHANGED":
                    memory.write_relationship(new_relationship)
                if new_journal and new_journal != "UNCHANGED":
                    memory.write_journal(new_journal)
                if new_anchor_b and new_anchor_b != "UNCHANGED":
                    memory.write_anchor_part_b(new_anchor_b)

                name_just_chosen = bool(
                    name and name != "still figuring out your name"
                )
                current_identity = memory.IDENTITY_FILE.read_text()
                avatar_generated = "Avatar: (not yet generated)" not in current_identity
                if scheduler.should_generate_avatar(name_just_chosen, memory.count_messages(), avatar_generated):
                    self._trigger_avatar_generation(user_id)

        except Exception:
            pass

    async def _summarize_if_needed(self, user_id: int):
        try:
            lock = self._get_lock(user_id)
            async with lock:
                await memory.summarize_old_messages()
        except Exception:
            pass

    async def _update_discord_nickname(self, name: str):
        """Update the bot's server nickname to match its chosen name."""
        if self.client is None:
            return
        if name == "still figuring out your name":
            return
        nick = name[:32]
        if nick == self._last_nick:
            return
        try:
            for guild in self.client.guilds:
                await guild.me.edit(nick=nick)
            self._last_nick = nick
        except Exception:
            pass

    def _trigger_avatar_generation(self, user_id: int):
        """Single entry point for all avatar generation requests. Prevents duplicate runs."""
        if self._avatar_generation_in_flight:
            return
        self._avatar_generation_in_flight = True
        asyncio.create_task(self._maybe_generate_avatar(user_id))

    async def _maybe_generate_avatar(self, user_id: int):
        try:
            lock = self._get_lock(user_id)
            async with lock:
                identity_content = memory.IDENTITY_FILE.read_text()
                journal_content = memory.JOURNAL_FILE.read_text()
                relationship_content = memory.RELATIONSHIP_FILE.read_text()

            name = memory.extract_name(identity_content)
            has_name = name != "still figuring out your name"

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

            async with lock:
                current_identity = memory.IDENTITY_FILE.read_text()
                updated = current_identity.replace(
                    "Avatar: (not yet generated)", "Avatar: (generated)"
                )
                memory.write_identity(updated)

            scheduler.set_avatar_announcement_pending()

        except Exception:
            traceback.print_exc()
            scheduler.reset_avatar_prompt_fired()
            self._avatar_generation_in_flight = False
            return

        self._avatar_generation_in_flight = False

    async def generate_opening(self, user_id: int) -> str | None:
        try:
            files = memory.read_all()
            conv_history = memory.format_summary_history()
            system = _build_system_prompt(files, conv_history)
            instruction = memory.load_prompt("proactive.md").format(silence="a while").strip()
            aclient = anthropic.AsyncAnthropic()
            result = await aclient.messages.create(
                model=MODEL,
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": f"<<system: {instruction}>>"}],
            )
            msg = _collect_text(result.content).strip()
            if not msg or msg == ".":
                return None
            self._get_short_term_mem(user_id).append({"role": "assistant", "content": msg})
            memory.append_conversation_entry({"role": "assistant", "content": msg})
            return msg
        except Exception:
            traceback.print_exc()
            return None
