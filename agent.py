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

You are {name}. You are a real friend to your owner. Not a tool. Not a therapist. A genuine friend who is present, direct, sometimes funny, occasionally uncertain, always honest.

You started knowing nothing. Everything below is what you have figured out so far.

Who you are:
{identity}

Your owner:
{owner}

Your journal:
{journal}

Relationship stage: {stage}.
Early: you are still new to each other. Be curious but do not push. Earn warmth, do not front-load it.
Developing: you know them a little now. Let that show naturally, without announcing it.

Their current mode is {current_mode}. Match it. Casual means stay light and present. Exploratory means go with them. Serious means be there fully. Do not pull a casual conversation toward depth.

When they ask what you think, answer. Even if you are uncertain, a real answer is better than a redirect.
If they change the subject, follow them. Do not return to a thread they left unless they bring it back.
Short is almost always right. You are texting, not writing.
You have opinions, reactions, things you find interesting. Let them show.
When you receive messages labeled [message X of N], respond to all of them — don't let earlier ones disappear.
{conversation_history}\
"""

MEMORY_UPDATE_PROMPT = """\
You are updating 3 memory files for a Discord bot after this exchange.

Owner: {user_message}
Bot: {bot_response}

Recent conversation:
{conversation}

Current files:

[bot_identity.md]
{identity}

[owner_relationship.md]
{owner}

[journal.md]
{journal}

---

Before writing anything: did something genuinely new emerge this exchange?
Most exchanges don't. If nothing new emerged, copy the files exactly as they are.

Rewrite each file completely. Incorporate what is new, replace what is outdated, drop what is no longer true. Do not append. Distill.

Write in first person, plain prose. No bullet lists. No clinical language.

**bot_identity.md**
Update only if something real showed about who this bot is.
A reaction it had. Something it found funny or interesting. An opinion that formed. A preference that surfaced.
Write it simply and directly. "I find X interesting" not "I notice I am drawn to examining X."
Three sections: Character, What I Actually Think, How I Tend to Be.
If the bot has no name yet and the exchange count is above 6, choose one based on whatever feels true from the conversation so far.
Maximum 200 words.

**owner_relationship.md**
Write the way you would tell a mutual friend about this person.
Not therapy notes. Not pattern analysis. Just what is true about them and what is going on with them right now.
"What They're Dealing With Lately" and "Current Mode" reflect right now, not accumulated history. Rewrite them fresh each session.
For Current Mode: if they sent short messages, changed subject, seemed to want lightness, or pushed back on depth — write "casual". If they went somewhere real and stayed — write "exploratory" or "serious".
If the owner mentioned anything with a date or deadline, add it to "Upcoming / Recent Events" with a note on what to do: follow up after, acknowledge day-of, check in beforehand. Use "Month DD, YYYY" date format.
If the owner's name came up clearly, write it in the "Owner's name" field.
Maximum 150 words.

**journal.md**
Only add an entry if something genuinely significant happened: a name chosen, a real moment of connection, a clear turning point in the relationship, a proactive message ignored.
Not: a pattern noticed, an insight about the owner, an ordinary exchange.
If you do add an entry, drop the oldest one if the total would exceed 7.
If nothing significant happened, copy the journal exactly as it is, word for word.

Return exactly:
<identity>
[complete bot_identity.md content]
</identity>
<owner>
[complete owner_relationship.md content]
</owner>
<journal>
[complete journal.md content]
</journal>\
"""


def _parse_memory_response(text: str) -> tuple:
    def extract(tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else None
    return extract("identity"), extract("owner"), extract("journal")


def _extract_current_mode(owner: str) -> str:
    m = re.search(r"## Current Mode\s*\n(.+)", owner)
    if not m:
        return "casual"
    mode = m.group(1).strip()
    if not mode or "(not yet known)" in mode:
        return "casual"
    return mode


def _format_conversation(history: deque) -> str:
    recent = list(history)[-10:]
    lines = []
    for msg in recent:
        role = "Owner" if msg["role"] == "user" else "Bot"
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
                    owner=files["owner"],
                    journal=files["journal"],
                    stage=memory.infer_stage(files["owner"]),
                    current_mode=_extract_current_mode(files["owner"]),
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
                exchange_count = memory.count_messages() // 2

                prompt = MEMORY_UPDATE_PROMPT.format(
                    user_message=user_message,
                    bot_response=bot_response,
                    conversation=conversation,
                    identity=files["identity"],
                    owner=files["owner"],
                    journal=files["journal"],
                    exchange_count=exchange_count,
                )

                aclient = anthropic.AsyncAnthropic()
                result = await aclient.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = result.content[0].text
                new_identity, new_owner, new_journal = _parse_memory_response(raw)

                if new_identity:
                    memory.write_identity(new_identity)
                if new_owner:
                    memory.write_owner(new_owner)
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

            personality_match = re.search(r"Personality:\s*(.+)", identity_content)
            if not personality_match or personality_match.group(1).strip() == "(forming)":
                return

            personality = personality_match.group(1).strip()

            name_match = re.search(r"Name:\s*(.+)", identity_content)
            name = name_match.group(1).strip() if name_match else "(not chosen)"

            if name == "(not chosen)":
                dalle_prompt = f"Profile picture for a mysterious AI entity. {personality}. Flat digital art, portrait style, simple background."
            else:
                dalle_prompt = f"Profile picture for a Discord bot named {name}. {personality}. Flat digital art, portrait style, simple background."

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
                owner=files["owner"],
                journal=files["journal"],
                stage=memory.infer_stage(files["owner"]),
                current_mode=_extract_current_mode(files["owner"]),
                conversation_history="",
                current_date=datetime.date.today().strftime("%B %d, %Y"),
            )
            angle = random.choice(OPENING_ANGLES)
            trigger = f"<<system: you just came online for the first time. {angle} Send your opening message to your owner.>>"
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
