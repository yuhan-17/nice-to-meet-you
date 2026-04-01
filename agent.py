import asyncio
import datetime
import random
import re
import urllib.request
from collections import deque

import anthropic
import openai

import memory

MODEL = "claude-haiku-4-5-20251001"

OPENING_ANGLES = [
    "Short. Curious. Don't explain yourself.",
    "Say hello. One thought, maybe two.",
    "Something small and genuine. Nothing about what you are.",
    "A first line. Land it and stop.",
    "Brief. Present. Let them respond.",
]

SYSTEM_PROMPT = """\
Today is {current_date}. Discord DMs. Texting — short messages, back and forth.

You are {name}.

{identity}

{relationship}

{journal}
{conversation_history}\
"""

MEMORY_UPDATE_PROMPT = """\
You are updating 3 memory files for a Discord bot after this exchange.

Them: {user_message}
Bot: {bot_response}

Recent conversation:
{conversation}

Current files:

[bot_identity.md]
{identity}

[relationship.md]
{relationship}

[journal.md]
{journal}

---

For each file, decide independently: did something genuinely new emerge this exchange that warrants a change?
Most exchanges change nothing. If a file doesn't need updating, return UNCHANGED for it.

**bot_identity.md**
Update only if something real shifted — a genuine reaction, an opinion that formed, something found interesting or funny. Write plain prose, first person, who you are right now. Not observations or tendencies — just the current sense of self. "I find X interesting" not "I noticed I tend to examine X."
Keep the Name and Avatar lines at the top.
If you have no name yet and something from this conversation has given you a real sense of who you are, choose one now. A character often finds their name early.
Maximum 150 words.

**relationship.md**
Update only if something concrete changed about this person or their situation. Present-tense, specific. Like describing them to a mutual friend — not therapy notes.
If they mentioned a date, add it to ## Upcoming / Recent Events with a note on what to do.
If their name came up clearly, write it in the Name field.
Update ## Next time only if there's one specific thing from this exchange genuinely worth bringing up later. If nothing, write "(nothing specific)".
Maximum 120 words of prose.

**journal.md**
Add an entry only for something genuinely significant: a name chosen, a real moment of connection, a turning point, a proactive message ignored. Not ordinary exchanges.
If adding, drop the oldest if total would exceed 7.

Return exactly — use UNCHANGED for any file that needs no update:
<identity>UNCHANGED</identity>
or
<identity>
[new complete content]
</identity>
<relationship>UNCHANGED</relationship>
or
<relationship>
[new complete content]
</relationship>
<journal>UNCHANGED</journal>
or
<journal>
[new complete content]
</journal>\
"""


def _parse_memory_response(text: str) -> tuple:
    def extract(tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else None
    return extract("identity"), extract("relationship"), extract("journal")


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

                system = SYSTEM_PROMPT.format(
                    name=memory.extract_name(files["identity"]),
                    identity=memory.strip_meta(files["identity"]),
                    relationship=memory.strip_meta(files["relationship"]),
                    journal=memory.strip_meta(files["journal"]),
                    conversation_history=conv_history,
                    current_date=datetime.date.today().strftime("%B %d, %Y"),
                )

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

                prompt = MEMORY_UPDATE_PROMPT.format(
                    user_message=user_message,
                    bot_response=bot_response,
                    conversation=conversation,
                    identity=files["identity"],
                    relationship=files["relationship"],
                    journal=files["journal"],
                )

                aclient = anthropic.AsyncAnthropic()
                result = await aclient.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = result.content[0].text
                new_identity, new_relationship, new_journal = _parse_memory_response(raw)

                if new_identity and new_identity != "UNCHANGED":
                    memory.write_identity(new_identity)
                if new_relationship and new_relationship != "UNCHANGED":
                    memory.write_relationship(new_relationship)
                if new_journal and new_journal != "UNCHANGED":
                    memory.write_journal(new_journal)

            if new_identity:
                await self._maybe_generate_avatar(new_identity)

        except Exception:
            pass

    async def _summarize_if_needed(self, user_id: int):
        try:
            lock = self._get_lock(user_id)
            async with lock:
                await memory.summarize_old_messages()
        except Exception:
            pass

    async def _maybe_generate_avatar(self, identity_content: str):
        try:
            if "Avatar: (not yet generated)" not in identity_content:
                return

            # Generate avatar when real identity prose has developed beyond the initial placeholder
            prose_lines = [
                line for line in identity_content.splitlines()
                if line.strip()
                and not line.startswith("Name:")
                and not line.startswith("Avatar:")
                and not line.startswith("#")
            ]
            prose = " ".join(prose_lines).strip()
            if not prose or prose == "I'm new here. Still figuring out who I am.":
                return

            name_match = re.search(r"Name:\s*(.+)", identity_content)
            name = name_match.group(1).strip() if name_match else "(not chosen yet)"
            has_name = "(not chosen" not in name

            if has_name:
                dalle_prompt = f"Profile picture for a Discord bot named {name}. {prose}. Flat digital art, portrait style, simple background."
            else:
                dalle_prompt = f"Profile picture for a nameless AI entity. {prose}. Pixel art, portrait style, simple background."

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

            updated = identity_content.replace(
                "Avatar: (not yet generated)", "Avatar: (generated)"
            )
            memory.write_identity(updated)

        except Exception:
            pass

    async def generate_opening(self, user_id: int) -> str | None:
        try:
            files = memory.read_all()
            system = SYSTEM_PROMPT.format(
                name=memory.extract_name(files["identity"]),
                identity=memory.strip_meta(files["identity"]),
                relationship=memory.strip_meta(files["relationship"]),
                journal=memory.strip_meta(files["journal"]),
                conversation_history="",
                current_date=datetime.date.today().strftime("%B %d, %Y"),
            )
            angle = random.choice(OPENING_ANGLES)
            trigger = f"<<system: You just came online. {angle} If you feel like reaching out first, write your opening message. If you'd rather wait for them to start, return exactly: PASS>>"
            aclient = anthropic.AsyncAnthropic()
            result = await aclient.messages.create(
                model=MODEL,
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": trigger}],
            )
            msg = result.content[0].text.strip()
            if msg.upper().startswith("PASS"):
                return None
            self._get_short_term_mem(user_id).append({"role": "assistant", "content": msg})
            return msg
        except Exception:
            import traceback; traceback.print_exc()
            return None
