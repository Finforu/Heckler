#!/usr/bin/env bash
# Heckler installer for Linux and Windows (WSL2).
#
#   ./install.sh            install, set up .env, run the self-check
#   ./install.sh --dev      also install the test tools
#   ./install.sh --yes      don't ask (install uv if missing, keep defaults)
#
# Safe to run again: it only adds what's missing and never overwrites .env.
set -euo pipefail
cd "$(dirname "$0")"

DEV=0
YES=0
for arg in "$@"; do
  case "$arg" in
    --dev) DEV=1 ;;
    --yes|-y) YES=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (see --help)"; exit 2 ;;
  esac
done

bold=$(tput bold 2>/dev/null || true); dim=$(tput dim 2>/dev/null || true); reset=$(tput sgr0 2>/dev/null || true)
step() { echo; echo "${bold}==> $*${reset}"; }
note() { echo "    $*"; }
ask() {  # ask "Question?" -> 0 for yes
  [ "$YES" = 1 ] && return 0
  read -r -p "    $1 [Y/n] " answer </dev/tty || return 0
  [[ -z "$answer" || "$answer" =~ ^[Yy] ]]
}

# ── Where are we? ────────────────────────────────────────────────────────────
case "$(uname -s)" in
  Linux) ;;
  MINGW*|MSYS*|CYGWIN*)
    echo "Heckler runs on Windows through WSL2, not Git Bash or Cygwin."
    echo "In PowerShell (as admin):  wsl --install -d Ubuntu"
    echo "Then open Ubuntu, clone the repo inside it (e.g. in ~/) and run ./install.sh there."
    exit 1 ;;
  Darwin)
    echo "macOS isn't supported (no NVIDIA GPU). The bot could only run listen-only (VOICE=0)."
    ask "Continue anyway?" || exit 1 ;;
  *) echo "Unsupported system: $(uname -s)"; exit 1 ;;
esac
if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
  note "Running in WSL2."
  case "$PWD" in
    /mnt/*) note "${bold}Tip:${reset} you're under /mnt (the Windows drive), which is slow from WSL."
            note "Cloning into your Linux home (cd ~ && git clone ...) makes installs and model loading much faster." ;;
  esac
fi

# ── uv (Python + packages) ───────────────────────────────────────────────────
step "Checking for uv (installs Python 3.12 and the packages)"
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  note "uv isn't installed. It's a fast Python package manager: https://docs.astral.sh/uv/"
  if ask "Install it now (into ~/.local/bin)?"; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "Install uv, then run ./install.sh again."; exit 1
  fi
fi
note "$(uv --version)"

# ── Virtual environment ──────────────────────────────────────────────────────
step "Setting up Python 3.12 in .venv"
if [ -x .venv/bin/python ] && .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
  note "Already there."
else
  uv venv --python 3.12 .venv
fi

# ── Packages ─────────────────────────────────────────────────────────────────
GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  GPU=1
  note "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  note "No NVIDIA GPU found. ${dim}(WSL2: install the NVIDIA driver on Windows; nothing inside WSL.)${reset}"
fi
step "Installing packages (the first time downloads ~3 GB, mostly PyTorch)"
requirements=requirements.txt
[ "$DEV" = 1 ] && requirements=requirements-dev.txt
uv pip install --python .venv/bin/python -r "$requirements"

# ── .env ─────────────────────────────────────────────────────────────────────
step "Configuration (.env)"
if [ -f .env ]; then
  note ".env already exists: left as it is."
else
  cp .env.example .env
  chmod 600 .env
  note "Created .env from .env.example."
  echo
  note "Paste your bot token (Discord Developer Portal → your application → Bot → Reset Token)."
  note "It isn't shown while you type. Leave empty to add it to .env later."
  token=""
  if [ "$YES" != 1 ]; then read -r -s -p "    Token: " token </dev/tty || true; echo; fi
  if [ "$GPU" = 0 ]; then
    note "Without a GPU the bot can listen and run voice commands, but not talk."
  fi
  .venv/bin/python - "$token" "$GPU" <<'PY'
import sys
from pathlib import Path
token, gpu = sys.argv[1].strip(), sys.argv[2] == "1"
path = Path(".env")
lines = path.read_text(encoding="utf-8").splitlines()
lines = [f"DISCORD_BOT_TOKEN={token}" if line.startswith("DISCORD_BOT_TOKEN=") else line for line in lines]
if not gpu:
    lines += ["", "# No NVIDIA GPU found by install.sh: listen-only, speech-to-text on the CPU",
              "VOICE=0", "STT_ENGINE=parakeet"]
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
  [ -n "$token" ] && note "Token saved." || note "No token yet: put it after DISCORD_BOT_TOKEN= in .env."
fi

# ── Self-check ───────────────────────────────────────────────────────────────
step "Self-check"
set +e
.venv/bin/python doctor.py
status=$?
set -e

echo
if [ "$status" = 0 ]; then
  echo "${bold}Ready.${reset} Next:"
  echo "  1. Invite the bot with the link above (if you haven't)."
  echo "  2. Start it:   .venv/bin/python bot.py"
  echo "     The first start downloads the models (a few minutes). The log then prints"
  echo "     a dashboard login link; open it in your browser (from Windows too, with WSL2)."
  echo "  3. In Discord: join a voice channel and type /join."
else
  echo "Fix the items marked ✗ above, then check again with:  .venv/bin/python doctor.py"
fi
exit "$status"
