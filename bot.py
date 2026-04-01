import asyncio
import logging
import os

import discord
from dotenv import load_dotenv

import memory
import scheduler
from agent import Agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

OWNER_ID = int(os.getenv("OWNER_ID"))
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)

memory.ensure_files_exist()
agent = Agent(client=bot)

_pending: dict = {}       # user_id -> list of (content, ref_content or None)
_debounce_tasks: dict = {}
_last_message: dict = {}  # user_id -> last Discord message object


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    scheduler.start(bot, agent, OWNER_ID, CHANNEL_ID)

    files = memory.read_all()
    if "## Personality Hypotheses\n(none yet)" in files["owner"]:
        opening = await agent.generate_opening(OWNER_ID)
        channel = bot.get_channel(CHANNEL_ID)
        await channel.send(opening)


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if message.author.id != OWNER_ID:
        return

    scheduler.record_owner_message()

    ref_content = None
    if message.reference:
        try:
            ref_msg = await message.channel.fetch_message(message.reference.message_id)
            ref_content = ref_msg.content
        except Exception:
            pass

    uid = message.author.id
    channel = message.channel

    if uid not in _pending:
        _pending[uid] = []
    _pending[uid].append((message.content, ref_content))
    _last_message[uid] = message

    if uid in _debounce_tasks and not _debounce_tasks[uid].done():
        _debounce_tasks[uid].cancel()

    async def flush(uid=uid, channel=channel):
        await asyncio.sleep(3.5)
        messages = _pending.pop(uid, [])
        reply_to = _last_message.pop(uid, None)
        if not messages:
            return

        if len(messages) == 1:
            content, ref = messages[0]
            combined = f"[replying to: {ref}]\n{content}" if ref else content
        else:
            parts = []
            for content, ref in messages:
                if ref:
                    parts.append(f"[replying to: {ref}]\n{content}")
                else:
                    parts.append(content)
            combined = "\n---\n".join(parts)

        async with channel.typing():
            response = await agent.respond(uid, combined)

        response_parts = [p.strip() for p in response.split("||") if p.strip()]
        for i, part in enumerate(response_parts):
            if i > 0:
                await asyncio.sleep(1.5)
                await channel.send(part)
            else:
                await reply_to.reply(part)

    _debounce_tasks[uid] = asyncio.create_task(flush())


bot.run(os.getenv("DISCORD_TOKEN"))
