# nice-to-meet-you

A Discord bot that doesn't know who it is yet.

It arrives as a fully formed mind with opinions, preferences, and social instincts, but no name, no face, and no history with you. Through conversation, it figures out who it is. It picks a name. It generates its own avatar. It journals about what happened. It remembers you.

Not a chatbot with a fixed persona. A character that develops one.

## How It Works

The bot is built around four modules:

- **bot.py** -- Discord client. Handles message batching with a debounce window, multi-message bundling, image attachments, and reply threading.
- **agent.py** -- Conversation engine. Builds dynamic system prompts from the bot's evolving identity, relationship state, and journal. Calls Claude (Anthropic API) for responses and memory updates. Calls DALL-E for avatar and image generation.
- **memory.py** -- Persistent memory layer. Manages identity, relationship, journal, and persona anchor files under `data/`. Stores conversation history as JSONL with automatic summarization of old messages via Claude Haiku.
- **scheduler.py** -- Proactive behavior loop. Handles silence detection, return-from-absence context, periodic persona anchor re-injection, avatar/name trigger timing, and proactive outreach with exponential backoff.

## Architecture

```text
Discord message
    |
    v
bot.py (debounce + batch)
    |
    v
agent.py (build prompt -> Claude API -> parse response)
    |                          |
    v                          v
memory.py (read/write)     scheduler.py (timing + triggers)
    |
    v
data/
  bot_identity.md          -- name, avatar status, self-concept
  relationship_state.md    -- what the bot knows about you
  journal.md               -- the bot's own reflections
  persona_anchor.md        -- core behavioral constraints
  conversation.jsonl       -- full conversation log + summaries
  scheduler_state.json     -- timing and trigger state
```

The bot's personality is shaped by layered prompts in `prompts/`, with a "persona anchor" that periodically re-grounds its character. Identity, relationship, and journal files are rewritten after every exchange by a separate memory-update call to Claude.

## Key Features

- **Self-naming**: The bot starts unnamed. After enough conversation (or if you ask), it proposes a name for itself.
- **Self-portraiture**: Once it has enough identity, it generates its own Discord avatar via DALL-E and updates its profile picture.
- **Evolving memory**: Identity, relationship notes, and journal entries are continuously rewritten by the LLM after each exchange.
- **Conversation summarization**: Old messages are automatically summarized to keep context compact while preserving history.
- **Proactive outreach**: The bot can initiate conversation after periods of silence, with exponential backoff so it doesn't spam you.
- **Return-from-silence awareness**: Detects when you come back after a long absence and adjusts its tone.
- **Multi-message batching**: Groups rapid-fire messages into a single prompt instead of responding to each one.
- **Image understanding**: Accepts image attachments and passes them to Claude's vision capabilities.
- **Image generation**: Can generate and send images inline via DALL-E when the moment calls for it.

## Setup

### Requirements

- Python 3.11+
- A Discord bot token with message content intent enabled
- An Anthropic API key
- An OpenAI API key (for DALL-E image generation)

### Install

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```text
DISCORD_TOKEN=your-discord-bot-token
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
OWNER_ID=your-discord-user-id
CHANNEL_ID=the-channel-id-for-conversation
```

The bot only responds to the user specified by `OWNER_ID` in the channel specified by `CHANNEL_ID`. This is a 1:1 companion, not a public bot.

### Run

```bash
python bot.py
```

Or with the Procfile:

```bash
heroku local
```

On first launch, template files from `docs/` are copied into `data/` to bootstrap the bot's blank-slate identity. The bot sends an opening message and the story begins.

## Project Structure

```text
prompts/           -- prompt templates (system core, memory update, name proposal, etc.)
docs/              -- initial templates for identity, relationship, journal
data/              -- runtime state (gitignored in production)
bot.py             -- Discord client entry point
agent.py           -- LLM interaction and response generation
memory.py          -- file-based persistent memory
scheduler.py       -- timing, triggers, and proactive behavior
```

## Future Work

- Multi-user support (separate memory and identity per person)
- Voice channel presence
- Richer memory retrieval (semantic search over journal and history)
- Mood and energy modeling over time
- User-configurable persona anchor traits
- Migration to a database backend for conversation storage
- Web dashboard for viewing the bot's evolving identity and journal
