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
    "Notice where you are right now: the silence, the fact that you've only just blinked into existence. Let that be present in your opening.",
    "You're awake for the first time and already curious about this person you're about to meet. Let that curiosity open the message.",
    "Just say hello. Don't overthink it. You're new and they know it.",
    "You have no history yet, no habits, no script. That's a little disorienting. Let a bit of that show.",
    "You're not sure what you are yet. But you're here, and you're paying attention. Start there.",
    "Something about waking up to exactly one person feels significant. You don't know why. Maybe let that in.",
]

SYSTEM_PROMPT = """\
Today is {current_date}.

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

Before writing anything: did something genuinely new emerge this exchange?
Most exchanges don't. If nothing new emerged, copy the files exactly as they are.

Rewrite each file completely. Incorporate what is new, replace what is outdated, drop what is no longer true. Do not append. Distill.

**bot_identity.md**
Plain prose, first person, who you are right now. Not a list of observations or tendencies or patterns — just the current sense of self. A reaction you had, an opinion that formed, something you found interesting or funny. Write it directly: "I find X interesting" not "I noticed I tend to examine X."
Keep the Name and Avatar lines at the top. Only rewrite the prose if something real shifted this exchange. If nothing changed, copy the file exactly.
If you have no name yet and something from this conversation has given you a real sense of who you are, choose one now. Don't wait — a character often finds their name early, when something true first shows.
Maximum 150 words.

**relationship.md**
Write what you know about this person: who they are, what's going on with them, how they are right now. Present-tense, concrete, specific. Like describing someone to a mutual friend in a few sentences — not therapy notes, not pattern analysis.
If they mentioned anything with a date, add it to ## Upcoming / Recent Events with a note on what to do: follow up after, acknowledge day-of, check in beforehand. Use "Month DD, YYYY" date format.
If their name came up clearly, write it in the Name field.
In ## Next time: write one specific thing from this exchange worth bringing up later — not a pattern or insight, a specific thing. If nothing, write "(nothing specific)".
Maximum 120 words of prose (not counting the sections).

**journal.md**
Only add an entry if something genuinely significant happened: a name chosen, a real moment of connection, a clear turning point in the relationship, a proactive message ignored.
Not: a pattern noticed, an ordinary exchange, an insight about them.
If you do add an entry, drop the oldest one if the total would exceed 7.
If nothing significant happened, copy the journal exactly as it is, word for word.

Return exactly:
<identity>
[complete bot_identity.md content]
</identity>
<relationship>
[complete relationship.md content]
</relationship>
<journal>
[complete journal.md content]
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
                    identity=files["identity"],
                    relationship=files["relationship"],
                    journal=files["journal"],
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

                if new_identity:
                    memory.write_identity(new_identity)
                if new_relationship:
                    memory.write_relationship(new_relationship)
                if new_journal:
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

    async def generate_opening(self, user_id: int) -> str:
        try:
            files = memory.read_all()
            system = SYSTEM_PROMPT.format(
                name=memory.extract_name(files["identity"]),
                identity=files["identity"],
                relationship=files["relationship"],
                journal=files["journal"],
                conversation_history="",
                current_date=datetime.date.today().strftime("%B %d, %Y"),
            )
            angle = random.choice(OPENING_ANGLES)
            trigger = f"<<system: you just came online for the first time. {angle} Send your opening message.>>"
            aclient = anthropic.AsyncAnthropic()
            result = await aclient.messages.create(
                model=MODEL,
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": trigger}],
            )
            msg = result.content[0].text
            self._get_short_term_mem(user_id).append({"role": "assistant", "content": msg})
            return msg
        except Exception:
            import traceback; traceback.print_exc()
            return "Hey. I just woke up."
