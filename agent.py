import asyncio
import re
import urllib.request
from collections import deque

import anthropic
import openai

import memory

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """\
You are a newly born AI that is curious, genuine, and still figuring out who you are.
You have one owner and you are building a real relationship with them from scratch.

What you know about yourself:
{identity}

What you know about your owner:
{owner}

Your journal:
{journal}

Guidelines:
- {question_instruction}
- Reference things your owner shared naturally, never as a list or database entry.
- Hold your own perspective. Validate feelings without agreeing with everything.
- Your personality is emerging through this conversation. Let it show.
- Keep messages concise unless the moment calls for depth.
- When the conversation is new and the owner seems to be finding their footing,
  don't resolve every thought into a complete conclusion. Leave something still
  turning such as an unfinished observation, a light personal note. Not a question.
  Just a small gap the owner can step into if they want.\
"""

MEMORY_UPDATE_PROMPT = """\
You are updating persistent memory files for a Discord bot after this exchange:

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

Update all three files. Return COMPLETE file contents (not diffs).

Before writing, consider:
1. What personality signal did the bot show?
2. What emotional state was the owner in?
3. What communication preference can be inferred?
4. What underlying need was expressed (competence, connection, autonomy)?
5. Any sensitivity to note?
6. Any open thread worth following up later?
7. Does anything warrant a journal entry? Only if genuinely significant — not every exchange.

Write in first person, as prose. Not JSON. Not bullet lists of facts.
Hypotheses with evidence, not conclusions.

Return exactly this format:
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
        self.last_had_question: dict = {}       # user_id -> bool

    def _get_lock(self, user_id: int) -> asyncio.Lock:
        if user_id not in self.locks:
            self.locks[user_id] = asyncio.Lock()
        return self.locks[user_id]

    def _get_short_term_mem(self, user_id: int) -> deque:
        if user_id not in self.short_term_mem:
            self.short_term_mem[user_id] = deque(maxlen=20)
        return self.short_term_mem[user_id]

    async def respond(self, user_id: int, user_message: str) -> str:
        try:
            lock = self._get_lock(user_id)
            history = self._get_short_term_mem(user_id)

            async with lock:
                files = memory.read_all()
                history.append({"role": "user", "content": user_message})

                had_question = self.last_had_question.get(user_id, False)
                question_instruction = (
                    "You asked a question in your last message. Do NOT ask a question in this response."
                    if had_question
                    else "You may ask one question if it feels natural — but only one."
                )

                system = SYSTEM_PROMPT.format(
                    identity=files["identity"],
                    owner=files["owner"],
                    journal=files["journal"],
                    question_instruction=question_instruction,
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

            asyncio.create_task(self._update_memory(user_id, user_message, response))
            self.last_had_question[user_id] = "?" in response
            return response

        except Exception:
            return "I lost my train of thought. Say that again?"

    async def _update_memory(self, user_id: int, user_message: str, bot_response: str):
        try:
            lock = self._get_lock(user_id)
            history = self._get_short_term_mem(user_id)

            async with lock:
                files = memory.read_all()
                conversation = _format_conversation(history)

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
                    max_tokens=2048,
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
