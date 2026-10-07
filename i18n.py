"""Everything the bot says in text (replies, consent messages, notices) in
the server's language.

Strings live in locales/<language>/*.json: flat {"key": "text"} files, all
files of a language merged (so different parts of the code keep their own
file). Missing keys fall back to English, then to the key itself, so a
half-translated language still works.

    t("consent.voice.ask", "es", bot="Heckler")

A server's language is its "language" setting (dashboard / /language), else
DEFAULT_LANGUAGE from .env, else English. It also picks the speech-to-text
language and the starter pack.
"""
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

LOCALES = Path(__file__).resolve().parent / "locales"
FALLBACK = "en"


class _Keep(dict):
    """format_map helper: unknown {placeholders} are left as they are."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@lru_cache(maxsize=None)
def _strings(lang: str) -> dict[str, str]:
    merged: dict[str, str] = {}
    folder = LOCALES / lang
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            merged.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            log.warning("Skipping broken locale file %s: %s", path, e)
    return merged


def reload() -> None:
    """Re-read the locale files (after editing them)."""
    _strings.cache_clear()


def languages() -> list[str]:
    """The languages there are strings for, e.g. ["en", "es"]."""
    return sorted(p.name for p in LOCALES.iterdir() if p.is_dir()) if LOCALES.is_dir() else [FALLBACK]


def default_language() -> str:
    lang = os.getenv("DEFAULT_LANGUAGE", "").strip().lower()
    return lang if lang in languages() else FALLBACK


def guild_language(store, guild_id: int | None) -> str:
    """The server's language: its "language" setting, else the default."""
    lang = store.get_setting(guild_id, "language") if guild_id is not None else None
    return lang if lang in languages() else default_language()


def t(key: str, lang: str | None = None, **values) -> str:
    """The text for `key` in `lang` (English, then the key, when missing),
    with {placeholders} filled from `values`."""
    for candidate in (lang or default_language(), FALLBACK):
        text = _strings(candidate).get(key)
        if text is not None:
            return text.format_map(_Keep(values))
    log.debug("Missing string %r", key)
    return key
