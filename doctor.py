"""Heckler self-check: is everything in place to run the bot?

    .venv/bin/python doctor.py

Checks Python, the installed packages, the GPU, disk space, the .env file
and (online) that the Discord token works, that the Message Content intent
is on, and prints the invite link. Exit code 1 if something must be fixed.
Only uses the standard library until it checks the packages themselves.
"""
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import warnings
from pathlib import Path

# Old packages (pydub) print syntax warnings on import: noise here.
warnings.filterwarnings("ignore", category=SyntaxWarning)

ROOT = Path(__file__).resolve().parent
INVITE = "https://discord.com/oauth2/authorize?client_id={}&scope=bot+applications.commands&permissions=3214336"
# Discord application flags: the Message Content intent, approved or "limited" (under 100 servers).
MESSAGE_CONTENT = (1 << 18) | (1 << 19)
MODELS_GB = 6  # Whisper + OmniVoice (+ Parakeet) on first run

problems = 0
color = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def say(mark: str, text: str, hint: str = "") -> None:
    global problems
    colors = {"ok": "32", "warn": "33", "fail": "31"}
    symbol = {"ok": "✓", "warn": "!", "fail": "✗"}[mark]
    if mark == "fail":
        problems += 1
    prefix = f"\033[{colors[mark]}m{symbol}\033[0m" if color else symbol
    print(f"  {prefix} {text}")
    if hint:
        print(f"      → {hint}")


def section(title: str) -> None:
    print(f"\n{title}")


def env() -> dict[str, str]:
    """.env as a dict (without exporting it)."""
    values = {}
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.split(" #", 1)[0].strip().strip('"').strip("'")
    return values


def check_system() -> None:
    section("System")
    v = sys.version_info
    if v[:2] == (3, 12):
        say("ok", f"Python {platform.python_version()}")
    elif v >= (3, 10):
        say("warn", f"Python {platform.python_version()}", "Heckler is developed and tested on Python 3.12")
    else:
        say("fail", f"Python {platform.python_version()}", "Python 3.12 is needed (install.sh sets it up with uv)")
    system = platform.system()
    release = platform.release().lower()
    if system == "Windows":
        say("fail", "Running on Windows directly", "use WSL2 (Ubuntu), see the README")
    elif "microsoft" in release or "wsl" in release:
        say("ok", "WSL2")
    else:
        say("ok", f"{system} {platform.machine()}")


def check_gpu(settings: dict[str, str]) -> None:
    section("GPU")
    voice_on = settings.get("VOICE", "1") == "1"
    stt = settings.get("STT_ENGINE", "whisper").lower()
    smi = shutil.which("nvidia-smi")
    if not smi:
        needs = voice_on or stt in ("whisper", "hybrid")
        say("warn" if not needs else "fail", "No NVIDIA GPU found (nvidia-smi)",
            "for a CPU-only install set VOICE=0 and STT_ENGINE=parakeet in .env" if needs else "")
        return
    try:
        out = subprocess.run([smi, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        name, total, used = [x.strip() for x in out.split(",")]
        total_gb, free_gb = int(total) / 1024, (int(total) - int(used)) / 1024
        needed = (2.3 if voice_on else 0) + (2.3 if stt in ("whisper", "hybrid") else 0)
        if total_gb + 0.5 < needed + 1.5:
            say("warn", f"{name}, {total_gb:.1f} GB", f"tight for this setup (~{needed + 1.5:.0f} GB): "
                "try STT_ENGINE=parakeet (speech-to-text on the CPU)")
        else:
            say("ok", f"{name}, {total_gb:.1f} GB ({free_gb:.1f} GB free now)")
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as e:
        say("warn", f"nvidia-smi didn't answer: {e}")
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            say("ok", f"PyTorch {torch.__version__} sees the GPU")
        else:
            say("fail" if voice_on else "warn", f"PyTorch {torch.__version__} can't use the GPU",
                "update the NVIDIA driver (WSL2: install it on Windows, not inside WSL)")
    except ImportError:
        pass  # reported under Packages


def check_packages(settings: dict[str, str]) -> None:
    section("Packages")
    needed = ["discord", "discord.ext.voice_recv", "davey", "dotenv", "numpy", "soxr", "soundfile", "yaml",
              "aiohttp", "av", "faster_whisper"]
    if settings.get("VOICE", "1") == "1":
        needed += ["torch", "torchaudio", "transformers", "pydub"]
    if settings.get("STT_ENGINE", "whisper").lower() in ("parakeet", "hybrid"):
        needed.append("sherpa_onnx")
    missing = []
    for name in needed:
        try:
            importlib.import_module(name)
        except Exception as e:  # ImportError, or a broken native library
            missing.append(f"{name} ({type(e).__name__})")
    if missing:
        say("fail", "Missing or broken: " + ", ".join(missing), "run ./install.sh again")
    else:
        say("ok", f"All {len(needed)} needed packages import")


def check_disk() -> None:
    section("Disk")
    cache = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    probe = cache if cache.exists() else Path.home()
    free_gb = shutil.disk_usage(probe).free / 1e9
    have_models = (cache / "hub" / "models--k2-fsa--OmniVoice").exists()
    if have_models:
        say("ok", f"Models already downloaded ({free_gb:.0f} GB free)")
    elif free_gb < MODELS_GB + 2:
        say("fail", f"Only {free_gb:.1f} GB free", f"the first run downloads ~{MODELS_GB} GB of models to {cache}")
    else:
        say("ok", f"{free_gb:.0f} GB free (the first run downloads ~{MODELS_GB} GB of models)")
    data = ROOT / "data"
    try:
        data.mkdir(exist_ok=True)
        (data / ".write-test").write_text("ok")
        (data / ".write-test").unlink()
        say("ok", "data/ is writable")
    except OSError as e:
        say("fail", f"Can't write to data/: {e}")


def discord_get(path: str, token: str) -> dict:
    request = urllib.request.Request(f"https://discord.com/api/v10{path}", headers={
        "Authorization": f"Bot {token}", "User-Agent": "Heckler-doctor (https://github.com/Finforu/Heckler, 1)"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def check_discord(settings: dict[str, str], online: bool) -> None:
    section("Discord")
    if not (ROOT / ".env").is_file():
        say("fail", "No .env file", "copy .env.example to .env (install.sh does it) and put your bot token in it")
        return
    token = settings.get("DISCORD_BOT_TOKEN", "").strip()
    if token.lower().startswith("bot "):
        token = token.split(" ", 1)[1]
    if not token:
        say("fail", "DISCORD_BOT_TOKEN is empty in .env",
            "Discord Developer Portal → your application → Bot → Reset Token, then paste it")
        return
    if not online:
        say("ok", "A token is set (not checked: --offline)")
        return
    try:
        me = discord_get("/users/@me", token)
        app = discord_get("/applications/@me", token)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            say("fail", "Discord rejects the token", "reset it in the Developer Portal (Bot → Reset Token) and update .env")
        else:
            say("warn", f"Discord answered {e.code}; couldn't check the token")
        return
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        say("warn", f"Couldn't reach Discord to check the token: {e}")
        return
    say("ok", f"Token works: logged in as {me.get('username')} (id {me.get('id')})")
    if int(app.get("flags", 0)) & MESSAGE_CONTENT:
        say("ok", "Message Content intent is on")
    else:
        say("fail", "Message Content intent is off",
            "Developer Portal → your application → Bot → Privileged Gateway Intents → Message Content")
    print(f"\n  Invite link (needs Manage Server on the server you add it to):\n  {INVITE.format(app.get('id'))}")


def main() -> int:
    online = "--offline" not in sys.argv
    settings = env()
    print("Heckler self-check")
    check_system()
    check_packages(settings)
    check_gpu(settings)
    check_disk()
    check_discord(settings, online)
    print()
    if problems:
        print(f"{problems} thing(s) to fix before starting the bot.")
        return 1
    print("All good. Start the bot with:  .venv/bin/python bot.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
