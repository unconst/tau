#!/usr/bin/env bash
#
# Tau — one-command install
#
# Usage:
#   curl -fsSL https://tau.ninja/install.sh | bash
#   TAU_BOT_TOKEN=... CHUTES_API_TOKEN=... ./install.sh   # non-interactive
#

set -e
set -o pipefail

# ── Colors ───────────────────────────────────────────────────────────────────

if [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ]; then
    GREEN=$'\033[0;32m' RED=$'\033[0;31m' CYAN=$'\033[0;36m'
    BOLD=$'\033[1m' DIM=$'\033[2m' NC=$'\033[0m'
else
    GREEN='' RED='' CYAN='' BOLD='' DIM='' NC=''
fi

ok()  { printf "  ${GREEN}+${NC} %s\n" "$1"; }
err() { printf "  ${RED}x${NC} %s\n" "$1"; }
die() { err "$1"; exit 1; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

# ── Spinner ──────────────────────────────────────────────────────────────────

spin() {
    local pid=$1 msg="$2" i=0 chars='|/-\'
    printf "\033[?25l" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r  ${CYAN}%s${NC} %s" "${chars:$((i%4)):1}" "$msg"
        sleep 0.1 2>/dev/null || sleep 1
        i=$((i+1))
    done
    printf "\033[?25h" 2>/dev/null || true
    wait "$pid" 2>/dev/null; local code=$?
    if [ $code -eq 0 ]; then
        printf "\r  ${GREEN}+${NC} %s\n" "$msg"
    else
        printf "\r  ${RED}x${NC} %s\n" "$msg"
    fi
    return $code
}

run() {
    local msg="$1"; shift
    local tmp=$(mktemp)
    "$@" >"$tmp" 2>&1 &
    local pid=$!
    if ! spin $pid "$msg"; then
        tail -3 "$tmp" 2>/dev/null | while IFS= read -r l; do printf "    %s\n" "$l"; done
        rm -f "$tmp"
        return 1
    fi
    rm -f "$tmp"
}

# ── Detect context ───────────────────────────────────────────────────────────

REPO_URL="${TAU_REPO_URL:-https://github.com/unconst/tau.git}"
INSTALL_DIR=""
RUNNING_FROM_REPO=false

if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$script_dir/tauctl" ] && [ -d "$script_dir/tau" ]; then
        INSTALL_DIR="$script_dir"
        RUNNING_FROM_REPO=true
    fi
fi

# ── Banner ───────────────────────────────────────────────────────────────────

printf "\n${CYAN}${BOLD}"
printf "   ___________               _______  .__            __        \n"
printf "   \\__    ___/____   __ __   \\      \\ |__| ____     |__|____   \n"
printf "     |    |  \\__  \\ |  |  \\  /   |   \\|  |/    \\    |  \\__  \\  \n"
printf "     |    |   / __ \\|  |  / /    |    \\  |   |  \\   |  |/ __ \\_\n"
printf "     |____|  (____  /____/  \\____|__  /__|___|  /\\__|  (____  /\n"
printf "                  \\/                \\/        \\/\\______|    \\/ \n"
printf "${NC}\n"

# ── 1. Collect credentials ──────────────────────────────────────────────────

HAS_TTY=false
if [ -t 0 ] || { [ -e /dev/tty ] && (echo >/dev/tty) 2>/dev/null; }; then
    HAS_TTY=true
fi

# Bot token
BOT_TOKEN="${TAU_BOT_TOKEN:-}"
if [ -z "$BOT_TOKEN" ]; then
    if [ "$HAS_TTY" = true ]; then
        printf "  ${DIM}Create a bot via @BotFather -> /newbot${NC}\n\n"
        printf "  Bot token: "
        read -r BOT_TOKEN </dev/tty 2>/dev/null || true
    fi
    [ -z "$BOT_TOKEN" ] && die "Bot token required (set TAU_BOT_TOKEN or run interactively)"
fi

# Chutes API token
CHUTES_KEY="${CHUTES_API_TOKEN:-}"
if [ -z "$CHUTES_KEY" ]; then
    if [ "$HAS_TTY" = true ]; then
        printf "  ${DIM}Get a key at https://chutes.ai${NC}\n\n"
        printf "  Chutes API token: "
        read -r CHUTES_KEY </dev/tty 2>/dev/null || true
    fi
    [ -z "$CHUTES_KEY" ] && die "Chutes API token required (set CHUTES_API_TOKEN or run interactively)"
fi

# Resolve bot name for install dir
BOT_NAME=""
resp=$(curl -sf --max-time 10 "https://api.telegram.org/bot${BOT_TOKEN}/getMe" 2>/dev/null) || true
if [ -n "$resp" ]; then
    BOT_NAME=$(printf '%s' "$resp" | grep -o '"username":"[^"]*"' | head -1 | cut -d'"' -f4 | tr '[:upper:]' '[:lower:]')
fi
if [ -n "$BOT_NAME" ]; then
    ok "Bot: @$BOT_NAME"
fi

# Set install dir
if [ -z "$INSTALL_DIR" ]; then
    INSTALL_DIR="$HOME/${BOT_NAME:-tau}"
fi
printf "\n"

# ── 2. Install ──────────────────────────────────────────────────────────────

printf "${BOLD}  Installing to $INSTALL_DIR${NC}\n\n"

# Check prerequisites
for cmd in git python3 curl; do
    command_exists "$cmd" || die "$cmd is required but not found. Install it first."
done

# Install uv
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command_exists uv; then
    run "Installing uv" bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command_exists uv || die "uv install failed"
fi

# Clone or pull
if [ "$RUNNING_FROM_REPO" = true ]; then
    cd "$INSTALL_DIR"
    run "Pulling latest" git pull || true
elif [ -d "$INSTALL_DIR/tauctl" ] 2>/dev/null || [ -f "$INSTALL_DIR/tauctl" ]; then
    cd "$INSTALL_DIR"
    run "Pulling latest" git pull || true
else
    run "Cloning repository" git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Python environment
if [ ! -d ".venv" ]; then
    run "Creating environment" uv venv .venv
fi
source .venv/bin/activate

# Install packages
run "Installing agent" uv pip install -e ./agent
run "Installing tau" uv pip install -e .
run "Installing supervisor" uv pip install supervisor

# Create directories
mkdir -p context/logs

# Write .env
cat > .env << EOF
TAU_BOT_TOKEN=$BOT_TOKEN
CHUTES_API_TOKEN=$CHUTES_KEY
EOF
ok "Configuration saved"

# ── 3. Start ────────────────────────────────────────────────────────────────

INSTANCE_ID=$(basename "$INSTALL_DIR" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')

OS=$(uname -s)
if [ "$OS" = "Darwin" ]; then
    # macOS — use launchd
    PLIST_DIR="$HOME/Library/LaunchAgents"
    PLIST_NAME="com.tau.${INSTANCE_ID}"
    PLIST_FILE="$PLIST_DIR/$PLIST_NAME.plist"
    GUI_UID=$(id -u)

    mkdir -p "$PLIST_DIR"
    cat > "$PLIST_FILE" << PEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$PLIST_NAME</string>
    <key>ProgramArguments</key>
    <array>
        <string>$INSTALL_DIR/.venv/bin/supervisord</string>
        <string>-n</string>
        <string>-c</string>
        <string>$INSTALL_DIR/supervisord.conf</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$INSTALL_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>$INSTALL_DIR/context/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>$INSTALL_DIR/context/logs/launchd.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>$HOME</string>
        <key>PATH</key>
        <string>$INSTALL_DIR/.venv/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
PEOF

    launchctl bootout "gui/$GUI_UID/$PLIST_NAME" 2>/dev/null || \
        launchctl unload "$PLIST_FILE" 2>/dev/null || true
    sleep 1
    launchctl bootstrap "gui/$GUI_UID" "$PLIST_FILE" 2>/dev/null || \
        launchctl load "$PLIST_FILE" 2>/dev/null || true
    ok "Auto-start configured (launchd)"

elif [ "$OS" = "Linux" ] && systemctl --user status >/dev/null 2>&1; then
    # Linux — use systemd
    SERVICE_DIR="$HOME/.config/systemd/user"
    SERVICE_NAME="tau-${INSTANCE_ID}"
    mkdir -p "$SERVICE_DIR"
    cat > "$SERVICE_DIR/$SERVICE_NAME.service" << SEOF
[Unit]
Description=Tau ($INSTANCE_ID)
After=network.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
Environment="HOME=$HOME" "PATH=$INSTALL_DIR/.venv/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$INSTALL_DIR/.venv/bin/supervisord -n -c $INSTALL_DIR/supervisord.conf
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
SEOF
    systemctl --user daemon-reload
    systemctl --user enable "$SERVICE_NAME" >/dev/null 2>&1 || true
    systemctl --user start "$SERVICE_NAME" || true
    loginctl enable-linger "$USER" 2>/dev/null || true
    ok "Auto-start configured (systemd)"
fi

# Wait for startup
sleep 2
if [ -f ".supervisord.pid" ]; then
    pid=$(cat .supervisord.pid 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        ok "Supervisord running"
    fi
elif "$INSTALL_DIR/tauctl" start >/dev/null 2>&1; then
    ok "Started via tauctl"
fi

# ── Done ─────────────────────────────────────────────────────────────────────

printf "\n"
printf "  ${GREEN}${BOLD}Tau is live${NC}\n"
printf "\n"
printf "  ${BOLD}Message your bot to begin${NC}\n"
printf "\n"
