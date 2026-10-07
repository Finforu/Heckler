"""Optional LLM answers: "Heckler, what's the capital of France?"

A question after the wake word that isn't one of the server's voice
commands goes to an LLM, and the bot says the answer out loud. Off unless
LLM_PROVIDER is set in .env:

    LLM_PROVIDER   anthropic (the Claude API), openai, ollama or lmstudio
    LLM_MODEL      the model; anthropic defaults to claude-sonnet-5-5 (fast, and cheaper than
                   Opus for short spoken answers), the others need one
    LLM_BASE_URL   another server (default: the provider's usual address)
    LLM_API_KEY    the key (default: ANTHROPIC_API_KEY / OPENAI_API_KEY; local servers need none)
    LLM_TIMEOUT_S  give up after this long (default 20)

The Claude API goes through the official `anthropic` package. OpenAI,
Ollama and LM Studio all speak the OpenAI chat-completions API, called
directly with aiohttp (which discord.py already installs).

Only what's said to the bot after its name is sent, with the asker's
display name and the last few questions and answers in that server (so
"and how tall is it?" works). Answers are kept short and plain, because a
voice reads them.
"""
import logging
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from ipaddress import ip_address
from urllib.parse import urlparse

log = logging.getLogger(__name__)

PROVIDERS = {
    # name: (default base URL, default model, the environment variable with its key)
    "anthropic": (None, "claude-sonnet-5-5", "ANTHROPIC_API_KEY"),
    "openai": ("https://api.openai.com/v1", None, "OPENAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None, None),
    "lmstudio": ("http://localhost:1234/v1", None, None),
}
PROVIDER_NAMES = {"anthropic": "the Claude API (Anthropic)", "openai": "OpenAI", "ollama": "Ollama",
                  "lmstudio": "LM Studio"}
# Claude models that take an effort level: answers here are short, so "low" keeps them quick.
CLAUDE_EFFORT_MODELS = {"claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5",
                        "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6"}
# Claude models that can hand a declined request to a fallback model (Claude API only).
CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
LANGUAGE_NAMES = {"en": "English", "es": "Spanish"}
HISTORY_TURNS = 6         # questions and answers remembered per server
HISTORY_MINUTES = 10      # ... for this long
MAX_ANSWER_CHARS = 350    # longer answers are cut at a sentence end
MAX_QUESTION_CHARS = 500


@dataclass
class Config:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    timeout_s: float = 20.0

    @property
    def cloud(self) -> bool:
        """Do questions leave this machine (or its local network)?"""
        if self.base_url is None:
            return True
        host = (urlparse(self.base_url).hostname or "").lower()
        if host in ("localhost", "") or host.endswith((".local", ".lan", ".internal")):
            return False
        try:
            address = ip_address(host)
        except ValueError:
            return True
        return not (address.is_private or address.is_loopback or address.is_link_local)

    @property
    def service(self) -> str:
        """Who answers, for people: "the Claude API (Anthropic)", "Ollama"..."""
        return PROVIDER_NAMES.get(self.provider, self.provider)


def config_from_env(env=None) -> Config | None:
    """The LLM settings in .env, or None when there's no LLM. ValueError
    (in plain words) when they're set but unusable."""
    env = os.environ if env is None else env
    provider = env.get("LLM_PROVIDER", "").strip().lower()
    if provider in ("", "none", "off", "0"):
        return None
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER={provider!r}: use one of {', '.join(PROVIDERS)} (or leave it empty)")
    default_url, default_model, key_env = PROVIDERS[provider]
    model = env.get("LLM_MODEL", "").strip() or default_model
    if not model:
        raise ValueError(f"LLM_PROVIDER={provider} needs LLM_MODEL (the model's name on that server)")
    key = env.get("LLM_API_KEY", "").strip() or (env.get(key_env, "").strip() if key_env else "") or None
    if provider in ("anthropic", "openai") and not key:
        raise ValueError(f"LLM_PROVIDER={provider} needs an API key: set LLM_API_KEY (or {key_env})")
    try:
        timeout = float(env.get("LLM_TIMEOUT_S", "") or 20)
    except ValueError:
        raise ValueError("LLM_TIMEOUT_S must be a number of seconds") from None
    base_url = env.get("LLM_BASE_URL", "").strip().rstrip("/") or default_url
    return Config(provider, model, base_url, key, timeout)


def system_prompt(bot: str, language: str | None, persona: str = "", now: datetime | None = None) -> str:
    now = now or datetime.now().astimezone()
    lines = [
        f"You are {bot}, a bot in a Discord voice call with a group of friends. People talk to you out loud: "
        "speech-to-text turns what they say into the messages you get, each starting with the speaker's name, "
        "so a word may be misheard and your own name may still be in the message. "
        "A text-to-speech voice reads your answer aloud in the call.",
        f"Answer in {LANGUAGE_NAMES.get(language or 'en', language or 'English')}. Keep it short: one to three "
        "sentences, under 60 words. Write plain spoken sentences only, with no lists, markdown, emoji, links or "
        "code, and write numbers and symbols the way a person reads them out. If you don't know or can't help, "
        "say so in a few words.",
        f"It is {now:%A, %B %d, %Y, %H:%M} where the bot runs.",
    ]
    if persona and persona.strip():
        lines.append(f"Instructions from the server's admins: {persona.strip()}")
    return "\n\n".join(lines)


def clean(answer: str | None) -> str:
    """Something a voice can read: no reasoning tags, no markdown, one
    paragraph, at most MAX_ANSWER_CHARS (cut at a sentence end)."""
    text = re.sub(r"<think>.*?(</think>|$)", " ", answer or "", flags=re.S | re.I)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)      # [label](link) -> label
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"(^|\s)[-*•]\s+|[*_#`~>|]", " ", text)      # list bullets and markdown
    text = " ".join(text.split())
    if len(text) > MAX_ANSWER_CHARS:
        cut = text[:MAX_ANSWER_CHARS]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        text = cut[:end + 1] if end > 40 else cut.rsplit(" ", 1)[0] + "…"
    return text.strip()


class LLM:
    """One LLM, shared by every server. `client` replaces the Anthropic SDK
    client and `session` the aiohttp session (tests)."""

    def __init__(self, config: Config, *, client=None, session=None):
        self.config = config
        self._client = client
        self._session = session
        self._history: defaultdict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_TURNS))

    @property
    def label(self) -> str:
        return f"{self.config.provider}: {self.config.model}"

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        if self._client is not None and hasattr(self._client, "close"):
            try:
                await self._client.close()
            except Exception:
                pass

    def forget(self, guild_id: int | None = None) -> None:
        if guild_id is None:
            self._history.clear()
        else:
            self._history.pop(guild_id, None)

    def _recent(self, guild_id: int) -> list[dict]:
        cutoff = time.monotonic() - HISTORY_MINUTES * 60
        messages = []
        for asked_at, question, answer in self._history[guild_id]:
            if asked_at >= cutoff:
                messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        return messages

    async def ask(self, question: str, *, guild_id: int, speaker: str, bot: str, language: str | None = None,
                  persona: str = "") -> str | None:
        """The answer, ready to be said, or None (nothing usable came back:
        an error, a refusal, an empty answer). Never raises for API trouble."""
        question = f"{speaker}: {question.strip()[:MAX_QUESTION_CHARS]}"
        system = system_prompt(bot, language, persona)
        messages = [*self._recent(guild_id), {"role": "user", "content": question}]
        started = time.monotonic()
        try:
            if self.config.provider == "anthropic":
                raw = await self._ask_claude(system, messages)
            else:
                raw = await self._ask_openai_compatible(system, messages)
        except Exception as e:  # network, auth, rate limits, bad JSON: the bot just can't answer
            log.warning("The LLM (%s) didn't answer: %s: %s", self.label, type(e).__name__, e)
            return None
        answer = clean(raw)
        log.info("LLM answered in %.1fs: %s", time.monotonic() - started, answer or "(nothing)")
        if not answer:
            return None
        self._history[guild_id].append((time.monotonic(), question, answer))
        return answer

    # ------------------------------------------------------------ Claude
    def _claude(self):
        if self._client is None:
            from anthropic import AsyncAnthropic  # only needed with LLM_PROVIDER=anthropic

            self._client = AsyncAnthropic(api_key=self.config.api_key, base_url=self.config.base_url,
                                          timeout=self.config.timeout_s, max_retries=1)
        return self._client

    async def _ask_claude(self, system: str, messages: list[dict]) -> str | None:
        import anthropic

        model = self.config.model
        request = dict(model=model, max_tokens=2048, system=system, messages=messages)
        if model in CLAUDE_EFFORT_MODELS:
            request["output_config"] = {"effort": "low"}
        client = self._claude()
        try:
            if model in CLAUDE_FALLBACK_MODELS and self.config.base_url is None:
                # A request the model declines is retried on a fallback model in the same call.
                response = await client.beta.messages.create(**request, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                response = await client.messages.create(**request)
        except anthropic.AuthenticationError:
            log.warning("The Claude API rejected the key: check LLM_API_KEY / ANTHROPIC_API_KEY")
            return None
        except anthropic.NotFoundError:
            log.warning("The Claude API doesn't know the model %r: check LLM_MODEL", model)
            return None
        except anthropic.RateLimitError:
            log.warning("The Claude API is rate limiting this key; try again in a moment")
            return None
        if response.stop_reason == "refusal":
            log.info("The LLM declined to answer")
            return None
        return "".join(block.text for block in response.content if block.type == "text")

    # ------------------------------------------------------------ OpenAI-compatible
    async def _ask_openai_compatible(self, system: str, messages: list[dict]) -> str | None:
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout_s))
        headers = {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
        body = {"model": self.config.model, "messages": [{"role": "system", "content": system}, *messages],
                "max_tokens": 400, "temperature": 0.7, "stream": False}
        async with self._session.post(f"{self.config.base_url}/chat/completions", json=body, headers=headers) as r:
            if r.status >= 400:
                detail = (await r.text())[:300]
                log.warning("The LLM server (%s) answered %d: %s", self.config.base_url, r.status, detail)
                return None
            data = await r.json(content_type=None)
        choices = data.get("choices") or []
        return (choices[0].get("message") or {}).get("content") if choices else None
