"""The LLM adapter, with a fake Claude client and a fake OpenAI-compatible server."""
from types import SimpleNamespace

import pytest
from aiohttp import web

import llm
from llm import LLM, Config, clean, config_from_env, system_prompt

asyncio = pytest.mark.asyncio


# ------------------------------------------------------------ configuration
def test_config_from_env():
    assert config_from_env({}) is None
    assert config_from_env({"LLM_PROVIDER": "none"}) is None
    c = config_from_env({"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"})
    assert (c.provider, c.model, c.base_url, c.api_key, c.cloud) == ("anthropic", "claude-sonnet-5-5", None, "k", True)
    c = config_from_env({"LLM_PROVIDER": "Ollama", "LLM_MODEL": "llama3.2"})
    assert c.base_url == "http://localhost:11434/v1" and c.api_key is None and not c.cloud
    c = config_from_env({"LLM_PROVIDER": "lmstudio", "LLM_MODEL": "qwen", "LLM_BASE_URL": "http://192.168.1.20:1234/v1/"})
    assert c.base_url == "http://192.168.1.20:1234/v1" and not c.cloud
    c = config_from_env({"LLM_PROVIDER": "openai", "LLM_MODEL": "m", "LLM_API_KEY": "x", "LLM_TIMEOUT_S": "5"})
    assert c.cloud and c.timeout_s == 5 and c.service == "OpenAI"
    assert config_from_env({"LLM_PROVIDER": "ollama", "LLM_MODEL": "m", "LLM_BASE_URL": "https://llm.example.com/v1"}).cloud


@pytest.mark.parametrize("env, problem", [
    ({"LLM_PROVIDER": "skynet"}, "use one of"),
    ({"LLM_PROVIDER": "ollama"}, "needs LLM_MODEL"),
    ({"LLM_PROVIDER": "anthropic"}, "API key"),
    ({"LLM_PROVIDER": "openai", "LLM_MODEL": "m"}, "API key"),
    ({"LLM_PROVIDER": "ollama", "LLM_MODEL": "m", "LLM_TIMEOUT_S": "soon"}, "seconds"),
])
def test_bad_config(env, problem):
    with pytest.raises(ValueError, match=problem):
        config_from_env(env)


def test_prompt_and_clean():
    prompt = system_prompt("Heckler", "es", "Talk like a pirate.")
    assert "You are Heckler" in prompt and "Answer in Spanish" in prompt and "Talk like a pirate." in prompt
    assert "admins" not in system_prompt("Heckler", "en")
    assert clean("<think>hmm, let me see</think> **Paris** is the [capital](https://x.y).") == "Paris is the capital."
    assert clean("- one\n- two\n# Title") == "one two Title"
    long = "This is a sentence that goes on. " * 30
    cut = clean(long)
    assert len(cut) <= llm.MAX_ANSWER_CHARS and cut.endswith(".")
    assert clean(None) == "" and clean("<think>never finished") == ""


# ------------------------------------------------------------ Claude
class FakeMessages:
    def __init__(self, response=None, error=None):
        self.calls, self.response, self.error = [], response, error

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def claude_response(text="Paris.", stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="thinking", thinking=""),
                                                      SimpleNamespace(type="text", text=text)])


def fake_client(response):
    beta, plain = FakeMessages(response), FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=beta), messages=plain), beta, plain


@asyncio
async def test_claude_request_and_history():
    client, beta, plain = fake_client(claude_response("It's Paris."))
    bot = LLM(Config("anthropic", "claude-sonnet-5-5", None, "k"), client=client)
    assert await bot.ask("what's the capital of France", guild_id=1, speaker="Ana", bot="Heckler",
                         language="en") == "It's Paris."
    call = beta.calls[0]
    assert call["model"] == "claude-sonnet-5-5" and call["output_config"] == {"effort": "low"}
    assert call["betas"] == [llm.FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["messages"] == [{"role": "user", "content": "Ana: what's the capital of France"}]
    assert "Heckler" in call["system"] and not plain.calls

    await bot.ask("and how big is it", guild_id=1, speaker="Bo", bot="Heckler")
    assert [m["role"] for m in beta.calls[1]["messages"]] == ["user", "assistant", "user"]  # remembers
    await bot.ask("hi", guild_id=2, speaker="Cy", bot="Heckler")
    assert len(beta.calls[2]["messages"]) == 1  # per server
    bot.forget(1)
    await bot.ask("hi", guild_id=1, speaker="Ana", bot="Heckler")
    assert len(beta.calls[3]["messages"]) == 1


@asyncio
async def test_claude_other_models_and_refusals():
    client, beta, plain = fake_client(claude_response())
    bot = LLM(Config("anthropic", "claude-haiku-4-5", None, "k"), client=client)
    await bot.ask("hi", guild_id=1, speaker="Ana", bot="H")
    assert plain.calls and "output_config" not in plain.calls[0] and "fallbacks" not in plain.calls[0]
    client, beta, plain = fake_client(claude_response(stop="refusal"))
    bot = LLM(Config("anthropic", "claude-opus-5-5", None, "k"), client=client)
    assert await bot.ask("hi", guild_id=1, speaker="Ana", bot="H") is None


@asyncio
async def test_errors_never_raise():
    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(error=ConnectionError("down"))))
    bot = LLM(Config("anthropic", "claude-opus-5-5", None, "k"), client=client)
    assert await bot.ask("hi", guild_id=1, speaker="Ana", bot="H") is None


# ------------------------------------------------------------ OpenAI-compatible
@asyncio
async def test_openai_compatible(aiohttp_server):
    seen = []

    async def completions(request):
        seen.append((request.headers.get("Authorization"), await request.json()))
        if seen[-1][1]["model"] == "broken":
            return web.json_response({"error": "no such model"}, status=404)
        return web.json_response({"choices": [{"message": {"content": "<think>x</think>Four."}}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    server = await aiohttp_server(app)
    base = str(server.make_url("/v1"))

    bot = LLM(Config("ollama", "llama3.2", base, None))
    try:
        assert await bot.ask("two plus two", guild_id=1, speaker="Ana", bot="H", language="en") == "Four."
    finally:
        await bot.close()
    auth, body = seen[0]
    assert auth is None and body["model"] == "llama3.2" and body["stream"] is False
    assert body["messages"][0]["role"] == "system" and body["messages"][-1] == {"role": "user", "content": "Ana: two plus two"}

    bot = LLM(Config("openai", "broken", base, "sk-1"))
    try:
        assert await bot.ask("hi", guild_id=1, speaker="Ana", bot="H") is None
    finally:
        await bot.close()
    assert seen[-1][0] == "Bearer sk-1"
