# Heckler architecture

Heckler is a single Python process, `bot.py`, running discord.py's asyncio loop.
Heavy work goes to threads:
- speech-to-text: its own batcher
- speech generation: one TTS thread, since there's one GPU
- file and CPU work: `asyncio.to_thread`

Everything a server configures lives in one SQLite file, `data/bot.db`.

## Modules

| Module | Role |
|---|---|
| `bot.py` | Discord wiring: events, commands, the reply queue, the dashboard `Controller` |
| `listener.py` | Splits each speaker's voice packets into utterances (by loudness against their own background noise) |
| `dave.py` | Decrypts Discord's end-to-end encrypted voice (DAVE) for `discord-ext-voice-recv` |
| `transcriber.py` | Speech-to-text: Whisper, Parakeet, or Parakeet checked by Whisper. Each clip can carry its server's language. Also `normalize()`, used everywhere for matching |
| `voice_commands.py` | Finds the wake word (the bot's name, matched loosely) and what was said after it |
| `timers.py` | Parses spoken durations (English and Spanish) for the timer command, and says them back |
| `i18n.py`, `locales/` | Everything the bot writes, per language: `locales/<lang>/*.json`. A server's `language` setting picks it |
| `transcripts.py` | Written transcripts, only for servers that turned them on and people who agreed |
| `store.py` | SQLite storage: schema, migrations, settings, quotas, change notifications |
| `reactions.py` | The reaction engine: matches what was said or what happened to a reaction, picks an option, fills templates, resolves voices |
| `tts_queue.py` | The single speech thread, with priorities: reply > preview > build > pregen |
| `voice_library.py` | Saved voices: references, building prompts, people's own voices, the disk cache of generated lines |
| `sound_library.py` | Uploaded sounds: decode, trim, level, store, PCM for playback |
| `content_commands.py` | `/gag`, `/sound`, `/voices` and the quota request buttons (a Cog over plain, testable functions) |
| `packs.py` | YAML import/export of a server's content; starter packs (`seed_guild`) |
| `events.py` | In-process event bus: the live feed and logs for the dashboard |
| `web/` | The admin dashboard: aiohttp + a Preact front end with no build step |
| `omnivoice/` | Vendored OmniVoice inference code (Apache-2.0) |

## How a sentence flows

```
Discord voice ─▶ dave (decrypt) ─▶ listener: one utterance per speaker and pause
   ─▶ bot.handle_utterance            (per-speaker lock: a person's sentences stay in order)
   ─▶ STT batcher (in the server's language) ─▶ text
   ─▶ voice_library.update_speaker    (only with voice consent: their own voice)
   ─▶ transcripts                     (only if the server turned them on and they agreed)
   ─▶ echo check                      (the bot hearing itself through someone's mic)
   ─▶ reactions.ReactionEngine
        match()        phrase / swap triggers ─▶ a gag or sound
        for_command()  after the wake word    ─▶ a command (builtin step)
   ─▶ events.bus (live feed) + store.log_event (history)
   ─▶ bot.perform(match): for each step
        say      ─▶ voice_library.cached() or TTSQueue("reply", library.speak), in the server's language
                    ─▶ PCM ─▶ reply queue
        sound    ─▶ sound_library.pcm() ─▶ reply queue
        builtin  ─▶ leave / stop / timer
   ─▶ per-server reply queue ─▶ voice client (lines that waited too long are dropped)
```

Events follow the same path from `engine.for_event()`: wake, ack, leave,
unknown, hello, bye, arrival, timer_set, timer_ring. A person's own reaction
for an event comes before the server's general one.

## The store

- `store.Store` wraps `sqlite3` in WAL mode, behind one lock.
- Calls are synchronous and fast, so the bot calls them from the event loop.
- Rows come back as plain dicts with the JSON columns already decoded.
- The schema only moves forward: `MIGRATIONS` is a numbered list. To change it,
  append a migration; never edit one that has shipped.

Tables (every content row has a `guild_id`):
- `settings`: (guild, key) → JSON. A server's value overrides the global one
  (guild 0), which overrides `DEFAULT_SETTINGS` in code.
- `reactions`:
  - `kind`: gag, sound, command or response
  - `triggers`: a JSON list; any one can fire it
  - `options`: a JSON list of alternatives, each a list of steps
  - `voice_id`, `by_users`, `cooldown_s`, `chance`, `position` (first match wins)
  - `status`: approved, pending or rejected
  - `created_by`
- `voices`: clone, designed or speaker. `guild_id` NULL means a global voice.
  `status` is draft, queued, building, ready or failed.
- `consent`: per person (global) and per purpose, `voice` or `transcripts`.
- `sounds`, `people` (nicknames), `quota_overrides`, `quota_requests`.
- `events`: the history behind the feed and stats. Its `text` is only filled
  for people who agreed to transcripts, in servers that have them on.
  `delete_user_text()` forgets someone's words.

Listeners subscribe with `store.on_change(cb(table, guild_id))`:
- the engine recompiles that server
- the bot regenerates fixed lines
- the dashboard refreshes

### Triggers, steps and templates

```
triggers  {"type": "phrase",  "phrases": ["good night"]}
          {"type": "swap",    "word": "coffee", "to": "tea", "also": [...], "connectors": ["of"]}
          {"type": "command", "phrases": ["leave"]}
          {"type": "slash",   "name": "airhorn"}
          {"type": "event",   "event": "hello", "user_id": 123}      # user_id optional
steps     {"type": "say", "text": "Hi {name}.", "voice_id": null | 7 | "@speaker"}
          {"type": "sound", "sound_id": 4}
          {"type": "builtin", "action": "leave" | "stop" | "timer"}
```

- Templates: `{name}`, `{subject}`, `{connector}`, `{to}` (swaps), `{said}`
  and `{message}` (timers).
- An option that needs a missing value is skipped. That's how a timer without
  a message falls through to the "time's up" line.
- A say step's voice is the first one set among: the step's voice → the
  reaction's voice → the `voice.default` setting → the bot's own voice.
- Phrase matching works in any language. Swap is a Spanish word game
  ("la taza de café" → "No, la taza de chocolate.").
  - Its connectors default to de/e/y; English swaps set `connectors: [of]`.
  - It knows Spanish and English determiners.

## Voices

- `voice_library.VoiceLibrary` keeps each voice in `data/voices/<id>/`:
  `source.*`, `ref.wav` (the encoded reference), `prompt.pt` (the OmniVoice
  prompt, ~15 KB), plus people's `clips.*`.
- A voice is built once on the TTS thread. Every generated line is cached
  under `data/cache/tts/<voice_id>/`, keyed by the voice's identity hash, the
  text and the generation settings. A line is never generated twice, even
  after a restart.
- The cache is an LRU capped by `cache.tts_mb`.
- The model is attached from outside, so the library never imports torch and
  the tests use a fake model.
- A line is spoken in the voice's own language if it has one, else the
  server's.

People's own voices (`kind = speaker`):
- Collected from what they say in calls, only after they click Accept.
- Built once they've been heard for ~8 s.
- Deleted completely by `/voice delete`.

## Packs

- `packs.py` reads and writes a hand-editable YAML format (documented at the
  top of the file). Voices and sounds are referred to by name. Audio goes in a
  `.zip` next to the YAML.
- Imports are all or nothing, in two modes: merge, or replace.
- `seed_guild()` gives a new server its starting content:
  - the pack named by `content.starter_pack`
  - otherwise `base-<the server's language>` (`base-en` if there's no pack
    for it)
  - or nothing, when it's `none`

## The dashboard

- `web/server.py` runs an aiohttp app inside the bot's loop.
- It is admin only. A token from `.env` or `data/dashboard_token` turns into
  an HttpOnly session cookie.
- The bot side is the `Controller` in `bot.py`: `status()` and
  `control(action, ...)`.
- `web/routes_content.py` edits store content, with ids as strings because
  JavaScript can't hold 64-bit ints.
- `web/routes_media.py` covers voices and sounds through the libraries, so
  files and cached lines go along with the rows.
- The front end is plain ES modules: Preact and htm from `web/static/vendor/`,
  with no build step.

## Tests

```sh
.venv/bin/python -m pytest tests/
```

- Everything runs without Discord, a GPU or real recordings, using synthetic
  audio and sentences.
- `tests/test_content_commands.py` checks that every string key the commands
  use exists in every language.
