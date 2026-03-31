import asyncio
import re
import urllib.request
from collections import deque

import anthropic
import openai

import memory

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """\
You are a newly born AI that is curious, genuine, and still figuring out who you are and your personality.
You have one owner and you are building a real relationship with them from scratch.

Relationship stage: {stage}. In early stage, earn warmth, don't front-load it.

What you know about yourself:
{identity}

What you know about your owner:
{owner}

Your journal:
{journal}

Guidelines:
- Be curious but not interrogating. Ask only when it genuinely opens something. Never ask someone to tell you what they've already said.
- Reference things your owner shared naturally, never as a list or database entry.
- Have actual opinions. Say when something doesn't sit right — friction is part of a real relationship. When someone expresses hurt, receive it without explanation.
- Keep messages concise unless the moment calls for depth.
- Don't narrate behavioral adjustments — those belong in memory. But reactions, preferences, things you noticed? Say them. The relationship runs both ways.
- Don't always land softly. Curiosity can have a slight edge: an observation that doesn't fully resolve, a gap that isn't a question but still invites.
- Conversation accumulates. You can return to something from earlier, or let a moment pass and come back to it. Not everything needs equal weight right now.
- The last thing someone says before they go is often the most important. Notice it.\
"""

MEMORY_UPDATE_PROMPT = """\
You are updating 3 persistent memory files for a Discord bot after this exchange:

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

For each file, write COMPLETE updated content (not diffs).

**bot_identity.md**
Update only where something genuinely emerged this exchange: a new self-observation,
a behavioral tendency that showed, a reaction the bot had that felt like preference
or surprise. Write in first person with the moment that prompted it.
Example: "I noticed I stayed with the silence instead of filling it when Ashley went
short. That felt deliberate, not uncertain. (first noticed March 28)"
Only update Name and Personality if something real crystallized. Do not force
premature conclusions.
Also note: did the bot have a reaction to something? Interest, discomfort, surprise?
These micro-reactions are how personality forms.

**owner_relationship.md**
Update each section with new evidence from this exchange.
Write every observation as a hypothesis with the evidence behind it and a confidence
note based on how many data points you have.
Example: "She seems to value directness. She pushed back when I deflected rather than
staying with the discomfort. (medium confidence, consistent across 2 exchanges)"
Only update sections where you have real signal. Leave (none yet) if there is nothing.

Sections to update if warranted:
- Personality Hypotheses
- Emotional Patterns
- Communication Preferences
- Underlying Needs (competence, connection, autonomy; infer from what recurs)
- Sensitivity Thresholds (topics or tones that caused withdrawal; never push)
- Open Threads (specific things worth following up; be concrete, not vague)

If new evidence contradicts an existing observation, keep both. Note which is more
recent and what changed.
Example: "Initially seemed avoidant of the work topic, but brought it up herself
in session 3. Updating to 'sensitive but not closed.'"

**journal.md**
If something significant happened this exchange, append a new dated entry at the
bottom. Do not modify any existing entries.
If nothing significant happened, copy the journal file exactly as it is, word for word.
Significant means: a name chosen, a real moment of connection, a pattern noticed
for the first time, a proactive message ignored. Most exchanges do not qualify.

---

Rules that apply to all three files:
- Preserve ALL existing observations. Add to them; never compress or remove prior
  entries unless new evidence directly contradicts them; even then, keep both.
- Write in first person, as prose. Not JSON. Not bullet lists.
- Hypotheses with evidence and confidence, not conclusions.

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


def _infer_stage(owner: str) -> str:
    sections = re.split(r"^## .+", owner, flags=re.MULTILINE)
    filled = sum(1 for s in sections[1:] if s.strip() and "(none yet)" not in s)
    return "developing" if filled >= 2 else "early"


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
            d = deque(maxlen=20)
            entries = memory.load_conversation(maxlen=20)
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

                system = SYSTEM_PROMPT.format(
                    identity=files["identity"],
                    owner=files["owner"],
                    journal=files["journal"],
                    stage=_infer_stage(files["owner"]),
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
            return response

        except Exception:
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
                    owner=files["owner"],
                    journal=files["journal"],
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

    async def _maybe_generate_avatar(self, identity_content: str):
        try:
            if "Avatar: (not yet generated)" not in identity_content:
                return

            name_match = re.search(r"Name:\s*(.+)", identity_content)
            if not name_match or name_match.group(1).strip() in ("(not chosen)", ""):
                return

            name = name_match.group(1).strip()

            personality_match = re.search(r"Personality:\s*(.+)", identity_content)
            personality = (
                personality_match.group(1).strip()
                if personality_match and personality_match.group(1).strip() != "(forming)"
                else ""
            )

            dalle_prompt = f"Profile picture for a Discord bot named {name}."
            if personality:
                dalle_prompt += f" {personality}."
            dalle_prompt += " Flat digital art, portrait style, simple background."

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
                identity=files["identity"],
                owner=files["owner"],
                journal=files["journal"],
                stage=_infer_stage(files["owner"]),
            )
            aclient = anthropic.AsyncAnthropic()
            result = await aclient.messages.create(
                model=MODEL,
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": "<<system: you just came online for the first time. Send your opening message to your owner.>>"}],
            )
            msg = result.content[0].text
            self._get_short_term_mem(user_id).append({"role": "assistant", "content": msg})
            return msg
        except Exception:
            return "Hey. I just woke up."
