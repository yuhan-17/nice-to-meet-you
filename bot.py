import os

import discord
from dotenv import load_dotenv

import memory
import scheduler
from agent import Agent

load_dotenv()

OWNER_ID = int(os.getenv("OWNER_ID"))
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)

memory.ensure_files_exist()
agent = Agent(client=bot)


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

    async with message.channel.typing():
        response = await agent.respond(message.author.id, message.content)

    await message.channel.send(response)


bot.run(os.getenv("DISCORD_TOKEN"))
