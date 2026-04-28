#!/usr/bin/env bash
# =============================================================
# Nova Desktop Voice Assistant — One-Shot Setup Script
# Tested on Ubuntu 22.04+ / Debian 12+ / macOS 13+
# =============================================================

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Colour

log()  { echo -e "${BLUE}[nova]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# -----------------------------------------------------------
# 0. Validate Python version
# -----------------------------------------------------------
PY=$(python3 --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY" | cut -d. -f1)
PY_MINOR=$(echo "$PY" | cut -d. -f2)

[[ $PY_MAJOR -ge 3 && $PY_MINOR -ge 11 ]] || \
    die "Python 3.11+ required. Found: $PY"

log "Python $PY detected."

# -----------------------------------------------------------
# 1. System dependencies (Linux)
# -----------------------------------------------------------
if [[ "$(uname)" == "Linux" ]]; then
    log "Installing Linux system dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        portaudio19-dev \
        python3-pyaudio \
        wmctrl \
        xdotool \
        xclip \
        ffmpeg \
        curl \
        git
    ok "System packages installed."
elif [[ "$(uname)" == "Darwin" ]]; then
    log "Installing macOS dependencies via Homebrew..."
    command -v brew &>/dev/null || die "Homebrew not found. Install from https://brew.sh"
    brew install portaudio ffmpeg
    ok "Homebrew packages installed."
fi

# -----------------------------------------------------------
# 2. Python virtual environment
# -----------------------------------------------------------
if [[ ! -d .venv ]]; then
    log "Creating Python virtual environment..."
    python3 -m venv .venv
    ok "Virtual environment created at .venv/"
fi

source .venv/bin/activate
log "Virtual environment activated."

# -----------------------------------------------------------
# 3. Install Python dependencies
# -----------------------------------------------------------
log "Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
ok "Python dependencies installed."

# -----------------------------------------------------------
# 4. Playwright browsers
# -----------------------------------------------------------
log "Installing Playwright Chromium..."
playwright install chromium --with-deps
ok "Playwright Chromium installed."

# -----------------------------------------------------------
# 5. spaCy language model
# -----------------------------------------------------------
log "Downloading spaCy en_core_web_sm..."
python -m spacy download en_core_web_sm -q
ok "spaCy model installed."

# -----------------------------------------------------------
# 6. Create Nova home directory
# -----------------------------------------------------------
mkdir -p ~/.nova/plugins
log "Nova home directory: ~/.nova"

# -----------------------------------------------------------
# 7. Done
# -----------------------------------------------------------
echo ""
ok "Nova setup complete! Run 'source .venv/bin/activate && python -m nova.main' to start."
