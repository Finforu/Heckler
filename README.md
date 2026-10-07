# Heckler

A Discord voice bot that listens in voice channels and talks back. It
transcribes what people say. When someone says a trigger phrase, it answers
out loud with a gag, a sound or an action, in a cloned or designed voice.
Admins run it from a web dashboard, and members can add their own gags,
sounds and voices from Discord, within quotas.

- **Listens:** speech-to-text per speaker, with Whisper (GPU) or Parakeet (CPU).
- **Answers:**
  - gags: "phrase → reply", plus a "swap" word game
  - sound clips
  - voice commands after its name ("Hey Heckler, leave", timers)
  - greetings when people join or leave
- **Talks:** in [OmniVoice](https://github.com/k2-fsa/OmniVoice) voices:
  - its own voice, designed from a description or cloned from ~10 s of audio
  - voices cloned from samples you have permission to use
  - voices designed from a description ("female, elderly, low pitch")
  - with their consent, people's own voices

  Every voice is built once. Every line is generated once and cached on disk.
- **Speaks your language:** each server picks English or Spanish (`/language`).
  That sets the bot's text replies, the speech-to-text language and the
  starter content.
- **Dashboard:** status, controls, live feed, editors for reactions, voices,
  sounds, quotas and settings, a test bench, stats, and pack import/export.
- **Per server:** content, settings and quotas belong to each Discord server.

> **Heads-up:** Discord doesn't officially support bots *receiving* voice.
> Heckler uses community libraries (`discord-ext-voice-recv`, plus its own
> support for Discord's voice encryption, DAVE). A Discord change can break
> listening until those are updated.

## Requirements

- Python 3.12 and [ffmpeg](https://ffmpeg.org/) on the PATH. Developed on Linux.
- Hardware, roughly:

  | Setup | What you get |
  |---|---|
  | NVIDIA GPU with ~8 GB | Everything: Whisper (~2.3 GB) and OmniVoice (~2.3 GB, plus a ~1–2 GB burst while building a voice) |
  | Smaller GPU (~4–6 GB) | Talking, with `STT_ENGINE=parakeet` so speech-to-text runs on the CPU. Should fit; not tested |
  | CPU only | `VOICE=0 STT_ENGINE=parakeet`: listens, transcribes and runs voice commands, but doesn't talk. OmniVoice on a CPU works, but is too slow for live replies |

## Setup

1. **Create a Discord application** at <https://discord.com/developers/applications>.
   - Under **Bot**, create a bot and copy its token.
   - Under **Privileged Gateway Intents**, enable **Message Content**.
2. **Invite it** with this URL, replacing `CLIENT_ID` with your application's id:
   ```
   https://discord.com/oauth2/authorize?client_id=CLIENT_ID&scope=bot+applications.commands&permissions=3214336
   ```
   - The scopes are `bot` and `applications.commands` (slash commands).
   - The permissions are View Channels, Send Messages, Read Message History,
     Connect and Speak.
3. **Install:**
   ```sh
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python -r requirements.txt
   ```
   Plain `python -m venv` plus `pip install -r requirements.txt` works too.
4. **Configure:**
   ```sh
   cp .env.example .env
   ```
   Then set at least `DISCORD_BOT_TOKEN`. See [Configuring](#configuring).
5. **Run:**
   ```sh
   .venv/bin/python bot.py
   ```
   - The first start downloads the models from Hugging Face and builds the
     bot's voice, which takes a few minutes. Later starts take seconds.
   - The log prints a one-click **dashboard login link**
     (`http://127.0.0.1:8765` by default).
   - The token is saved in `data/dashboard_token`. Keep the dashboard on
     localhost, or put it behind HTTPS.

The bot joins a voice channel by itself when 2+ people are in it, or with
`/join`.

## Configuring

**`.env`** holds what's needed before Discord connects. `.env.example` lists
everything.

| Variable | Meaning |
|---|---|
| `BOT_NAME` | What people call the bot, and its wake word. Default `Heckler`. Pick a name of 5+ letters that doesn't sound like a common word: speech-to-text spells made-up names loosely, and the bot matches loosely too |
| `WAKE_WORDS` | Other names or spellings that also wake it, comma-separated |
| `BOT_VOICE_FILE` | Optional: ~10 s of clear speech to clone the bot's voice from |
| `BOT_VOICE_DESCRIPTION` | Used when there's no voice file. Default `male, young adult, moderate pitch`. Pick at most one of each (see below) |
| `DEFAULT_LANGUAGE` | `en` (default) or `es`, for servers that haven't picked one |
| `VOICE`, `STT_ENGINE`, `WHISPER_LANGUAGES`… | Speech settings, see `.env.example` |
| `DASHBOARD`, `DASHBOARD_HOST`, `DASHBOARD_PORT`, `DASHBOARD_TOKEN` | The admin dashboard |

The words a voice description (or `/voices create`) can use:

- gender: `male`, `female`
- age: `child`, `teenager`, `young adult`, `middle-aged`, `elderly`
- pitch: `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`,
  `very high pitch`
- style: `whisper`
- accent: `american`, `british`, `australian`, `canadian`, `indian`,
  `chinese`, `korean`, `japanese`, `portuguese`, `russian`, followed by
  `accent`

`/voices create` also understands them in Spanish ("mujer, mayor, grave").

**Everything else lives in the database** (`data/bot.db`) and is edited from
the dashboard. That includes reactions, voices, sounds, nicknames, cooldowns,
quotas, language and transcripts. Each server can override the global
settings.

- **Starter content:** a new server gets the starter pack for its language,
  `packs/base-en.yaml` or `packs/base-es.yaml`. It contains:
  - voice commands (leave, stop, timer)
  - stock replies and neutral greetings
  - a few example gags, turned off

  The `content.starter_pack` setting picks another pack, or `none`.
- **Packs** are YAML files you can read and edit by hand. Export a server's
  content from the dashboard or with
  `python packs.py export --guild <id> mine.yaml`, and import it elsewhere.
  The format is described at the top of `packs.py`.
- **User content.** Members create gags, sounds and voices with `/gag`,
  `/sound` and `/voices`.
  - Per-person quotas default to 15 gags, 15 sounds and 5 voices.
  - At the limit, a member can request more; admins approve or deny from
    Discord or the dashboard.
  - Set `user_content.needs_approval` to hold new content until an admin
    approves it.
  - Admins are the bot owner, members with *Manage Server*, or the role in
    `admin.role_id`.

## Commands

All of them work as slash commands and with the `!` prefix.

| Command | What it does |
|---|---|
| `/join`, `/leave` | Join your voice channel / leave |
| `/say <text>` | Say something in the call |
| `/timer <minutes> [text]`, `/timers [clear]` | Reminders |
| `/autojoin on\|off` | Join calls by itself |
| `/language [en\|es]` | The bot's language in this server (admins change it) |
| `/gag add` / `list` / `remove` / `test` | Your gags. `test` shows what a sentence would set off, and why |
| `/sound add` / `play` / `list` / `remove` | Upload sounds, play them, give them trigger phrases |
| `/voices list` / `preview` / `create` / `clone` / `remove` | Saved voices: designed from a description, or cloned from a sample you have permission to use |
| `/voice` (`optin` / `delete` / `refresh`) | Your own voice: consent, delete, rebuild |
| `/userclone on\|off` | Gags answer in the speaker's own voice (only people who agreed) |
| `/transcripts` (`on` / `off` / `optin` / `delete`) | Written transcripts: admins turn them on or off; you agree, or delete yours |
| `/nicknames` | Who's in the call and what the bot calls them |
| `/reload`, `/reset` | Re-read content; rejoin and clear stuck state |

Voice commands are the bot's name plus a phrase, for example "Heckler,
leave" or "Heckler, remind me in 10 minutes to stretch". Packs can change the
phrases.

## Privacy and consent

All data stays on the machine running the bot, under `data/` and
`transcripts/`. Both are in `.gitignore`. Never commit them or share them.

- **When the bot joins a call,** it posts a short notice: it's listening, and
  what it keeps.
- **Listening is not recording.**
  - Speech is transcribed in memory to react to it.
  - The event log behind the dashboard's feed and stats keeps what happened
    (who, which reaction), not the words.
- **Transcripts are off by default.**
  - An admin can turn them on with `/transcripts on`. The bot then announces
    it, and asks everyone to agree.
  - A person's words are written down (`transcripts/`, plus the event log)
    only after they click **Accept**.
  - `/transcripts delete` removes everything written down of what you said.
  - `/transcripts off` stops it for the server.
- **Voice cloning consent.**
  - Nobody's own voice is kept or cloned until they click **Accept** (`/voice`).
  - `/voice delete` removes their samples, their voice and every line
    generated with it.
- **Cloning others.**
  - `/voices clone` asks you to confirm you have permission to use the voice.
  - Cloning someone's voice without their permission is not allowed. The
    OmniVoice model card strictly prohibits unauthorized voice cloning,
    impersonation, fraud and scams.
- **Debug options** `SAVE_AUDIO` and `/record` write audio to disk. Leave
  them off unless you're testing.

## Licenses

- **Heckler** is licensed under the **GNU Affero General Public License v3.0**
  (see [LICENSE](LICENSE)). If you run a modified version for others, you
  must offer them its source.
- **Vendored code:** OmniVoice inference code in `omnivoice/` is Apache-2.0.
  Preact (MIT) and htm (Apache-2.0) are in `web/static/vendor/`. See
  [NOTICE.md](NOTICE.md).
- **Model weights** are downloaded at first run, not shipped here. Each has
  its own license, and those licenses apply to what you do with the bot:

  | Model | License |
  |---|---|
  | `k2-fsa/OmniVoice` | **CC-BY-NC**: non-commercial use only |
  | OmniVoice's audio tokenizer (Higgs Audio 2) | Boson Higgs Audio 2 Community License (based on the Meta Llama 3 Community License) |
  | `deepdml/faster-whisper-large-v3-turbo-ct2` (OpenAI Whisper) | MIT |
  | NVIDIA Parakeet TDT 0.6B v3 (sherpa-onnx int8) | CC-BY-4.0. Verify on the model page |

  **Because of the OmniVoice weights, a Heckler that talks may only be used
  non-commercially.**

  The Higgs Audio 2 license requires this attribution (verbatim from section
  1.b.i of the tokenizer's `LICENSE`):

  > Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson AI USA, Inc., All Rights Reserved and Meta Llama 3 licensed under the Meta Llama 3 Community License, Copyright Meta Platforms, Inc., All Right Reserved. based on Meta Llama 3

  > Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
  > Boson Higgs Audio 2 is licensed under the Boson Community License, Copyright © Boson AI USA, Inc. All Rights Reserved.

## Contributing

`docs/ARCHITECTURE.md` explains the modules and how a sentence travels through
the bot. Text the bot posts lives in `locales/<language>/*.json`; adding a
language starts there. Run the tests with:

```sh
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest tests/
```

## Support

<!-- TODO: Ko-fi / Buy Me a Coffee link -->
If Heckler makes your calls better, you can support its development here:
*(link coming soon)*.
