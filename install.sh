#!/usr/bin/env bash
#
# Tau Installation Script
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/unconst/tau/main/install.sh | bash
#   TAU_INSTALL_DIR=/path/to/tau curl -fsSL ... | bash
#   TAU_QUIET=1 ./install.sh
#

set -e

# ─────────────────────────────────────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────────────────────────────────────

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'
NC=$'\033[0m'
BOLD=$'\033[1m'
DIM=$'\033[2m'

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

QUIET=${TAU_QUIET:-0}
FAST=${TAU_FAST:-0}
REPO_URL="${TAU_REPO_URL:-https://github.com/unconst/tau.git}"

INSTALL_DIR=""
INSTANCE_ID=""
RUNNING_FROM_REPO=false
BOT_TOKEN=""

# ─────────────────────────────────────────────────────────────────────────────
# Output helpers — consistent icon semantics
#
#   ▸  Section header
#   →  Action / choice
#   ✓  Success
#   ·  Info
#   !  Warning
#   ✗  Error
# ─────────────────────────────────────────────────────────────────────────────

log() { [ "$QUIET" = "1" ] || printf "%s\n" "$1"; }

section()  { log ""; log "${BOLD}▸ $1${NC}"; log ""; }
action()   { log "  → $1"; }
ok()       { log "  ${GREEN}✓${NC} $1"; }
info()     { log "  ${DIM}·${NC} $1"; }
warn()     { log "  ${YELLOW}!${NC} $1"; }
err()      { log "  ${RED}✗${NC} $1"; }

check_ok()      { log "  ${GREEN}✓${NC} $1${DIM}${2:+ ($2)}${NC}"; }
check_missing() { log "  ${RED}✗${NC} $1"; }
check_pending() { log "  ${DIM}○${NC} $1 ${DIM}(will install)${NC}"; }

pause() {
    [ "$FAST" = "1" ] && return
    printf "\n  ${DIM}▸ Continue ⏎${NC} "
    read -r </dev/tty
}

prompt_yn() {
    [ "$FAST" = "1" ] && return 0
    printf "  → %s [Y/n]: " "$1"
    read -r -n 1 REPLY </dev/tty
    printf "\n"
    [[ ! $REPLY =~ ^[Nn]$ ]]
}

spinner() {
    local pid=$1
    local msg="$2"
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0
    
    printf "\033[?25l"
    while kill -0 "$pid" 2>/dev/null; do
        i=$(( (i+1) % 10 ))
        printf "\r  %s%s%s %s" "$CYAN" "${spin:$i:1}" "$NC" "$msg"
        sleep 0.1
    done
    printf "\033[?25h"
    
    wait "$pid"
    local code=$?
    
    if [ $code -eq 0 ]; then
        printf "\r  ${GREEN}✓${NC} %s\n" "$msg"
    else
        printf "\r  ${RED}✗${NC} %s\n" "$msg"
    fi
    return $code
}

run_spin() {
    local msg="$1"; shift
    "$@" >/dev/null 2>&1 &
    spinner $! "$msg"
}

command_exists() { command -v "$1" >/dev/null 2>&1; }

detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "macos";;
        Linux*)  echo "linux";;
        *)       echo "unknown";;
    esac
}

detect_linux_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif [ -f /etc/debian_version ]; then
        echo "debian"
    else
        echo "unknown"
    fi
}

generate_instance_id() {
    local path="$1"
    local base=$(basename "$path" | tr '[:upper:]' '[:lower:]' | \
                 sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//;s/-$//')
    
    if [ -z "$base" ] || [ "$base" = "tau" ]; then
        local hash=$(echo -n "$path" | md5 2>/dev/null || \
                     echo -n "$path" | md5sum | cut -d' ' -f1)
        base="${base:-tau}-${hash:0:6}"
    fi
    echo "$base"
}

detect_install_context() {
    local script_dir=""
    
    if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        
        if [ -f "$script_dir/tauctl" ] && [ -d "$script_dir/tau" ]; then
            INSTALL_DIR="$script_dir"
            RUNNING_FROM_REPO=true
            return 0
        fi
    fi
    
    RUNNING_FROM_REPO=false
    if [ -n "${TAU_INSTALL_DIR:-}" ]; then
        INSTALL_DIR="$TAU_INSTALL_DIR"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Installation steps
# ─────────────────────────────────────────────────────────────────────────────

step_welcome() {
    printf "\n"
    printf "%s%s" "$CYAN" "$BOLD"
    printf "   ████████╗ █████╗ ██╗   ██╗\n"
    printf "   ╚══██╔══╝██╔══██╗██║   ██║\n"
    printf "      ██║   ███████║██║   ██║\n"
    printf "      ██║   ██╔══██║██║   ██║\n"
    printf "      ██║   ██║  ██║╚██████╔╝\n"
    printf "      ╚═╝   ╚═╝  ╚═╝ ╚═════╝\n"
    printf "%s" "$NC"
    printf "   %sSelf-upgrading agent%s\n\n" "$DIM" "$NC"
    
    if [ -z "$INSTALL_DIR" ]; then
        if [ "$FAST" != "1" ]; then
            printf "Install path ${DIM}(default: ~/tau)${NC}: "
            read -r INSTALL_PATH </dev/tty
        else
            INSTALL_PATH=""
        fi
        
        if [ -z "$INSTALL_PATH" ]; then
            INSTALL_DIR="$HOME/tau"
        else
            INSTALL_DIR="${INSTALL_PATH/#\~/$HOME}"
        fi
        printf "\n"
    fi
    
    INSTANCE_ID=$(generate_instance_id "$INSTALL_DIR")
    
    if [ "$RUNNING_FROM_REPO" = true ]; then
        ok "Using existing repo"
    fi
    
    info "Installing to $INSTALL_DIR"
    
    if [ "$FAST" != "1" ]; then
        printf "\n  → Begin [Y/n]: "
        read -r -n 1 REPLY </dev/tty
        printf "\n"
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            exit 0
        fi
    fi
}

step_dependencies() {
    section "Dependencies"
    
    local OS=$(detect_os)
    local missing_git=false
    local missing_python=false
    local missing_curl=false
    
    if command_exists curl; then
        check_ok "curl"
    else
        check_missing "curl"
        missing_curl=true
    fi
    
    if command_exists git; then
        check_ok "git" "$(git --version | cut -d' ' -f3)"
    else
        check_missing "git"
        missing_git=true
    fi
    
    if command_exists python3; then
        check_ok "python3" "$(python3 --version 2>&1 | cut -d' ' -f2)"
    else
        check_missing "python3"
        missing_python=true
    fi
    
    if command_exists uv; then
        check_ok "uv"
    else
        check_pending "uv"
    fi
    
    if command_exists agent; then
        check_ok "cursor agent"
    else
        check_pending "cursor agent"
    fi
    
    if [ "$missing_git" = true ] || [ "$missing_python" = true ] || \
       [ "$missing_curl" = true ]; then
        log ""
        warn "Missing dependencies"
        
        if [ "$OS" = "macos" ]; then
            if prompt_yn "Install missing"; then
                install_macos_deps "$missing_git" "$missing_curl" "$missing_python"
            fi
        elif [ "$OS" = "linux" ]; then
            if prompt_yn "Install missing"; then
                install_linux_deps
            fi
        fi
        
        if ! command_exists git || ! command_exists python3 || \
           ! command_exists curl; then
            err "Missing: git, python3, curl"
            exit 1
        fi
    fi
    
    pause
}

install_macos_deps() {
    local need_git="$1"
    local need_curl="$2"
    local need_python="$3"
    
    if [ "$need_git" = true ] || [ "$need_curl" = true ]; then
        if ! xcode-select -p &>/dev/null; then
            info "Installing Xcode Command Line Tools"
            info "Complete the dialog, then press Enter"
            xcode-select --install 2>/dev/null || true
            pause
        fi
    fi
    
    if [ "$need_python" = true ]; then
        if ! command_exists brew; then
            if prompt_yn "Install Homebrew"; then
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                if [ -f "/opt/homebrew/bin/brew" ]; then
                    eval "$(/opt/homebrew/bin/brew shellenv)"
                fi
            fi
        fi
        
        if command_exists brew; then
            run_spin "Python installed" brew install python3
        fi
    fi
}

install_linux_deps() {
    local distro=$(detect_linux_distro)
    
    case "$distro" in
        ubuntu|debian|pop|linuxmint)
            sudo apt update >/dev/null 2>&1
            sudo apt install -y git python3 python3-pip python3-venv curl build-essential
            ;;
        fedora|rhel|centos|rocky)
            sudo dnf install -y git python3 python3-pip python3-devel curl gcc gcc-c++ make
            ;;
        arch|manjaro)
            sudo pacman -Sy --noconfirm git python python-pip curl base-devel
            ;;
        *)
            warn "Unknown distro: $distro"
            info "Install manually: git python3 python3-pip curl"
            ;;
    esac
}

step_install_uv() {
    command_exists uv && return
    
    (curl -LsSf https://astral.sh/uv/install.sh | sh) >/dev/null 2>&1 &
    spinner $! "uv installed"
    
    export PATH="$HOME/.local/bin:$PATH"
    
    if ! command_exists uv; then
        err "Failed to install uv"
        exit 1
    fi
}

step_clone() {
    section "Install"
    
    step_install_uv
    
    if [ "$RUNNING_FROM_REPO" = true ]; then
        if prompt_yn "Pull latest changes"; then
            cd "$INSTALL_DIR"
            git pull origin main >/dev/null 2>&1 &
            spinner $! "Pulled latest" || warn "Could not pull"
        fi
        
        cd "$INSTALL_DIR"
        return
    fi
    
    if [ -d "$INSTALL_DIR" ]; then
        if [ ! -f "$INSTALL_DIR/tauctl" ]; then
            cd "$INSTALL_DIR"
            run_spin "Repo updated" git pull origin main
        elif prompt_yn "Pull latest changes"; then
            cd "$INSTALL_DIR"
            git pull origin main >/dev/null 2>&1 &
            spinner $! "Pulled latest" || true
        fi
    else
        mkdir -p "$(dirname "$INSTALL_DIR")"
        run_spin "Repo cloned" git clone "$REPO_URL" "$INSTALL_DIR"
    fi
    
    cd "$INSTALL_DIR"
}

step_python() {
    cd "$INSTALL_DIR"
    
    if [ ! -d ".venv" ]; then
        run_spin "Environment created" uv venv .venv
    fi
    
    source .venv/bin/activate
    
    run_spin "Tau installed" uv pip install -e .
    run_spin "Supervisor installed" uv pip install supervisor
    
    mkdir -p "$INSTALL_DIR/logs"
    
    pause
}

step_cursor_agent() {
    command_exists agent && return
    
    log ""
    info "Cursor agent CLI required"
    
    if prompt_yn "Install Cursor CLI"; then
        curl https://cursor.com/install -fsSL | bash
        
        export PATH="$HOME/.cursor/bin:$HOME/.local/bin:$PATH"
        
        if command_exists agent; then
            ok "Cursor CLI installed"
            info "Opening browser for auth"
            log ""
            agent login
            ok "Authenticated"
        else
            warn "Installed but not in PATH"
            info "Restart terminal, then: agent login"
        fi
    else
        info "Add later: curl https://cursor.com/install -fsSL | bash"
    fi
}

step_telegram() {
    section "Telegram"
    
    if [ -n "$TAU_BOT_TOKEN" ]; then
        ok "Bot token detected"
        BOT_TOKEN="$TAU_BOT_TOKEN"
        if [ "$FAST" != "1" ] && ! prompt_yn "Use existing token"; then
            BOT_TOKEN=""
        fi
    fi
    
    if [ -z "$BOT_TOKEN" ]; then
        info "Create a bot via @BotFather → /newbot"
        log ""
        printf "  ${DIM}Token:${NC} "
        read -r BOT_TOKEN </dev/tty
        
        if [ -z "$BOT_TOKEN" ]; then
            err "Token required"
            if prompt_yn "Retry"; then
                step_telegram
                return
            fi
            exit 1
        fi
    fi
    
    echo "TAU_BOT_TOKEN=$BOT_TOKEN" > "$INSTALL_DIR/.env"
    ok "Token saved"
    
    pause
}

step_openai() {
    local ENV_FILE="$INSTALL_DIR/.env"
    
    section "Optional"
    
    if [ -n "$OPENAI_API_KEY" ]; then
        grep -q "OPENAI_API_KEY" "$ENV_FILE" 2>/dev/null || \
            echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> "$ENV_FILE"
        ok "OpenAI API key detected"
        return
    fi
    
    [ "$FAST" = "1" ] && return
    
    info "OpenAI key enables voice transcription"
    printf "  API key ${DIM}(Enter to skip)${NC}: "
    read -r API_KEY </dev/tty
    
    if [ -n "$API_KEY" ]; then
        echo "OPENAI_API_KEY=$API_KEY" >> "$ENV_FILE"
        ok "API key saved"
    fi
    :
}

setup_launchd() {
    local PLIST_DIR="$HOME/Library/LaunchAgents"
    local PLIST_NAME="com.tau.$INSTANCE_ID.supervisor"
    local PLIST_FILE="$PLIST_DIR/$PLIST_NAME.plist"
    
    mkdir -p "$PLIST_DIR"
    
    cat > "$PLIST_FILE" << EOF
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
    <true/>
    <key>StandardOutPath</key>
    <string>$INSTALL_DIR/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>$INSTALL_DIR/logs/launchd.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>$HOME</string>
        <key>PATH</key>
        <string>$INSTALL_DIR/.venv/bin:$HOME/.local/bin:$HOME/.cursor/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF
    
    launchctl unload "$PLIST_FILE" 2>/dev/null || true
    launchctl load "$PLIST_FILE"
    
    ok "Auto-start enabled"
}

setup_systemd() {
    local SERVICE_DIR="$HOME/.config/systemd/user"
    local SERVICE_NAME="tau-$INSTANCE_ID"
    local SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME.service"
    
    mkdir -p "$SERVICE_DIR"
    
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Tau ($INSTANCE_ID)
After=network.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/.venv/bin/supervisord -n -c $INSTALL_DIR/supervisord.conf
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF
    
    systemctl --user daemon-reload
    systemctl --user enable "$SERVICE_NAME" >/dev/null 2>&1
    systemctl --user start "$SERVICE_NAME"
    loginctl enable-linger "$USER" 2>/dev/null || true
    
    ok "Auto-start enabled"
}

step_launch() {
    section "Launch"
    
    # Auto-start option
    if prompt_yn "Enable auto-start"; then
        local OS=$(detect_os)
        
        if [ "$OS" = "macos" ]; then
            setup_launchd
        elif [ "$OS" = "linux" ]; then
            setup_systemd
        else
            warn "Auto-start not supported"
        fi
    fi
    
    # Check if already running
    if [ -f "$INSTALL_DIR/.supervisord.pid" ]; then
        local pid=$(cat "$INSTALL_DIR/.supervisord.pid" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            ok "Tau already running"
            print_final
            return
        fi
    fi
    
    if prompt_yn "Start Tau now"; then
        cd "$INSTALL_DIR"
        set -a; source .env; set +a
        
        "$INSTALL_DIR/tauctl" start >/dev/null 2>&1 &
        spinner $! "Tau started"
        
        print_final
    else
        log ""
        info "Start later: cd $INSTALL_DIR && ./tauctl start"
    fi
}

print_final() {
    log ""
    printf "  ${BOLD}${GREEN}✓ Tau is live${NC}\n"
    log ""
    info "Message your bot to begin"
    printf "  ${CYAN}This agent can modify itself.${NC}\n"
    log ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

detect_install_context

main() {
    if [[ -n "${1:-}" ]] && [[ "$1" != --* ]]; then
        INSTALL_DIR="${1/#\~/$HOME}"
        if [ -f "$INSTALL_DIR/tauctl" ] && [ -d "$INSTALL_DIR/tau" ]; then
            RUNNING_FROM_REPO=true
        else
            RUNNING_FROM_REPO=false
        fi
    fi
    
    step_welcome
    step_dependencies
    step_clone
    step_python
    step_cursor_agent
    step_telegram
    step_openai
    step_launch
}

main "$@"
