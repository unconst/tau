#!/usr/bin/env bash
#
# Tau Installation Script
#
# Usage:
#   curl -fsSL https://tau.ninja/install.sh | bash
#   TAU_INSTALL_DIR=/path/to/tau curl -fsSL ... | bash
#   TAU_QUIET=1 ./install.sh
#   TAU_DEBUG=1 ./install.sh  # verbose debug output
#   TAU_FAST=1 ./install.sh   # non-interactive, accept defaults
#

set -e
set -o pipefail  # Catch failures in pipes (curl | sh)

# ─────────────────────────────────────────────────────────────────────────────
# Environment Detection
# ─────────────────────────────────────────────────────────────────────────────

# Detect if we have a TTY for interactive prompts
HAS_TTY=false
if [ -t 0 ] || [ -e /dev/tty ]; then
    # Try to actually read from tty
    if (echo >/dev/tty) 2>/dev/null; then
        HAS_TTY=true
    fi
fi

# Detect terminal capabilities
HAS_COLOR=false
HAS_UNICODE=false
if [ -t 1 ]; then
    # Check for color support
    if [ -n "$TERM" ] && [ "$TERM" != "dumb" ]; then
        case "$TERM" in
            xterm*|rxvt*|screen*|tmux*|vt100*|linux*|ansi*|cygwin*)
                HAS_COLOR=true
                ;;
        esac
        # Also check COLORTERM
        if [ -n "$COLORTERM" ]; then
            HAS_COLOR=true
        fi
    fi
    # Check for Unicode (rough heuristic - check all common locale variables)
    if [ -n "$LANG" ] && [[ "$LANG" =~ [Uu][Tt][Ff]-?8 ]]; then
        HAS_UNICODE=true
    elif [ -n "$LC_ALL" ] && [[ "$LC_ALL" =~ [Uu][Tt][Ff]-?8 ]]; then
        HAS_UNICODE=true
    elif [ -n "$LC_CTYPE" ] && [[ "$LC_CTYPE" =~ [Uu][Tt][Ff]-?8 ]]; then
        HAS_UNICODE=true
    elif [ -n "$LC_MESSAGES" ] && [[ "$LC_MESSAGES" =~ [Uu][Tt][Ff]-?8 ]]; then
        HAS_UNICODE=true
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Colors (with fallback for non-color terminals)
# ─────────────────────────────────────────────────────────────────────────────

if [ "$HAS_COLOR" = true ]; then
    RED=$'\033[0;31m'
    GREEN=$'\033[0;32m'
    YELLOW=$'\033[1;33m'
    CYAN=$'\033[0;36m'
    NC=$'\033[0m'
    BOLD=$'\033[1m'
    DIM=$'\033[2m'
else
    RED=''
    GREEN=''
    YELLOW=''
    CYAN=''
    NC=''
    BOLD=''
    DIM=''
fi

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

QUIET=${TAU_QUIET:-0}
FAST=${TAU_FAST:-0}
DEBUG=${TAU_DEBUG:-0}
REPO_URL="${TAU_REPO_URL:-https://github.com/unconst/tau.git}"
SKIP_NET_CHECK=${TAU_SKIP_NET_CHECK:-0}
ALLOW_RM_OUTSIDE_HOME=${TAU_ALLOW_RM_OUTSIDE_HOME:-0}

# Force non-interactive if no TTY
if [ "$HAS_TTY" = false ]; then
    FAST=1
fi

INSTALL_DIR=""
INSTANCE_ID=""
RUNNING_FROM_REPO=false
BOT_TOKEN=""
BOT_NAME=""
CLEANUP_FILES=()  # Track files to clean up on failure
LOCK_DIR=""
LOCK_ACQUIRED=false

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup trap
# ─────────────────────────────────────────────────────────────────────────────

cleanup() {
    local exit_code=$?
    # Restore cursor if hidden
    printf "\033[?25h" 2>/dev/null || true
    # Release lock if acquired
    if [ "$LOCK_ACQUIRED" = true ] && [ -n "$LOCK_DIR" ]; then
        rm -rf "$LOCK_DIR" 2>/dev/null || true
    fi
    # Remove temp files
    for f in "${CLEANUP_FILES[@]}"; do
        rm -f "$f" 2>/dev/null || true
    done
    exit $exit_code
}
trap cleanup EXIT INT TERM

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
debug() { [ "$DEBUG" = "1" ] && printf "  ${DIM}[debug] %s${NC}\n" "$1" || true; }

# Use ASCII or Unicode symbols based on terminal capability
if [ "$HAS_UNICODE" = true ]; then
    SYM_SECTION="▸"
    SYM_OK="✓"
    SYM_WARN="!"
    SYM_ERR="✗"
    SYM_PENDING="○"
    SYM_ARROW="→"
else
    SYM_SECTION=">"
    SYM_OK="+"
    SYM_WARN="!"
    SYM_ERR="x"
    SYM_PENDING="o"
    SYM_ARROW="->"
fi

section()      { log ""; log "${BOLD}${SYM_SECTION} $1${NC}"; }
section_done() { printf "\r${BOLD}${SYM_SECTION} $1 ${GREEN}${SYM_OK}${NC}\n"; }
ok()           { log "  ${GREEN}${SYM_OK}${NC} $1"; }
info()         { log "  $1"; }
warn()         { log "  ${YELLOW}${SYM_WARN}${NC} $1"; }
err()          { log "  ${RED}${SYM_ERR}${NC} $1"; }

check_ok()      { log "  ${GREEN}${SYM_OK}${NC} $1${DIM}${2:+ ($2)}${NC}"; }
check_missing() { log "  ${RED}${SYM_ERR}${NC} $1"; }
check_pending() { log "  ${DIM}${SYM_PENDING}${NC} $1 ${DIM}(will install)${NC}"; }

pause() {
    [ "$FAST" = "1" ] && return
    if [ "$HAS_TTY" = true ]; then
        printf "\n${DIM}${SYM_SECTION} Continue [Enter]${NC} "
        read -r </dev/tty 2>/dev/null || true
    fi
}

prompt_yn() {
    [ "$FAST" = "1" ] && return 0
    if [ "$HAS_TTY" = false ]; then
        return 0  # Default to yes in non-interactive mode
    fi
    printf "  ${SYM_ARROW} %s [Y/n]: " "$1"
    read -r -n 1 REPLY </dev/tty 2>/dev/null || REPLY=""
    printf "\n"
    [[ ! $REPLY =~ ^[Nn]$ ]]
}

# Portable sleep - handles systems without fractional sleep support
portable_sleep() {
    local duration="$1"
    # Try fractional sleep first, fall back to 1 second minimum
    sleep "$duration" 2>/dev/null || sleep 1
}

spinner() {
    local pid=$1
    local msg="$2"
    local silent="${3:-false}"
    local i=0
    local interrupted=false
    
    # Preserve existing traps so we can restore them
    local prev_int_trap
    local prev_term_trap
    prev_int_trap=$(trap -p INT)
    prev_term_trap=$(trap -p TERM)
    
    # Signal handler to kill subprocess on Ctrl+C
    spinner_cleanup() {
        interrupted=true
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            # Give it a moment to terminate gracefully
            portable_sleep 0.5
            # Force kill if still running
            kill -9 "$pid" 2>/dev/null || true
        fi
        printf "\033[?25h" 2>/dev/null || true  # Show cursor
        printf "\r%*s\r" $((${#msg} + 6)) ""    # Clear line
    }
    
    # Set up signal handler
    trap spinner_cleanup INT TERM
    
    # Choose spinner characters based on Unicode support
    if [ "$HAS_UNICODE" = true ]; then
        local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
        local spin_len=10
    else
        local spin='|/-\'
        local spin_len=4
    fi
    
    # Hide cursor (with fallback)
    printf "\033[?25l" 2>/dev/null || true
    
    while kill -0 "$pid" 2>/dev/null && [ "$interrupted" = false ]; do
        i=$(( (i+1) % spin_len ))
        if [ "$HAS_UNICODE" = true ]; then
            # Each braille char is 3 bytes in UTF-8
            printf "\r  %s%s%s %s" "$CYAN" "${spin:$((i*3)):3}" "$NC" "$msg"
        else
            printf "\r  %s%s%s %s" "$CYAN" "${spin:$i:1}" "$NC" "$msg"
        fi
        portable_sleep 0.1
    done
    
    # Show cursor
    printf "\033[?25h" 2>/dev/null || true
    
    # Restore previous signal handling
    if [ -n "$prev_int_trap" ]; then
        eval "$prev_int_trap"
    else
        trap - INT
    fi
    if [ -n "$prev_term_trap" ]; then
        eval "$prev_term_trap"
    else
        trap - TERM
    fi
    
    # If interrupted, propagate the signal
    if [ "$interrupted" = true ]; then
        return 130  # Standard exit code for SIGINT
    fi
    
    wait "$pid" 2>/dev/null
    local code=$?
    
    if [ "$silent" = "true" ]; then
        printf "\r%*s\r" $((${#msg} + 6)) ""
    elif [ $code -eq 0 ]; then
        printf "\r  ${GREEN}${SYM_OK}${NC} %s\n" "$msg"
    else
        printf "\r  ${RED}${SYM_ERR}${NC} %s\n" "$msg"
    fi
    return $code
}

run_spin() {
    local msg="$1"; shift
    "$@" >/dev/null 2>&1 &
    spinner $! "$msg"
}

run_silent() {
    local msg="$1"; shift
    local tmplog=$(mktemp)
    CLEANUP_FILES+=("$tmplog")
    "$@" >"$tmplog" 2>&1 &
    local pid=$!
    spinner $pid "$msg" true
    local code=$?
    if [ $code -ne 0 ]; then
        printf "\r  ${RED}${SYM_ERR}${NC} %s\n" "$msg"
        if [ -s "$tmplog" ]; then
            # Show error output line by line for proper formatting
            info "Error output:"
            tail -5 "$tmplog" | head -3 | while IFS= read -r line; do
                info "  $line"
            done
        fi
    fi
    rm -f "$tmplog"
    return $code
}

command_exists() { command -v "$1" >/dev/null 2>&1; }

# Safe directory removal with validation to prevent catastrophic deletions
safe_rm_dir() {
    local dir="$1"
    
    # Validate the path is not empty
    if [ -z "$dir" ]; then
        err "safe_rm_dir: empty path provided"
        return 1
    fi
    
    # Prevent removal of critical paths
    case "$dir" in
        /|/bin|/boot|/dev|/etc|/home|/lib*|/opt|/proc|/root|/run|/sbin|/sys|/tmp|/usr|/var)
            err "safe_rm_dir: refusing to remove critical path: $dir"
            return 1
            ;;
        "$HOME")
            err "safe_rm_dir: refusing to remove home directory"
            return 1
            ;;
    esac
    
    # By default, refuse to remove anything outside HOME unless explicitly allowed
    if [ "$ALLOW_RM_OUTSIDE_HOME" != "1" ]; then
        case "$dir" in
            "$HOME"/*) ;;
            *)
                err "safe_rm_dir: refusing to remove path outside HOME: $dir"
                info "Set TAU_ALLOW_RM_OUTSIDE_HOME=1 to override"
                return 1
                ;;
        esac
    fi
    
    # Ensure it's actually a directory
    if [ ! -d "$dir" ]; then
        debug "safe_rm_dir: not a directory or doesn't exist: $dir"
        return 0
    fi
    
    # Perform the removal
    rm -rf "$dir"
}

# Portable in-place sed that works on both macOS (BSD) and Linux (GNU)
sed_inplace() {
    local pattern="$1"
    local file="$2"
    
    if [ "$(uname -s)" = "Darwin" ]; then
        # BSD sed requires empty string after -i
        sed -i '' "$pattern" "$file"
    else
        # GNU sed works with -i directly
        sed -i "$pattern" "$file"
    fi
}

# Safely load environment variables from .env file
# Only exports simple KEY=VALUE pairs, ignores comments and complex syntax
load_env_file() {
    local env_file="$1"
    
    if [ ! -f "$env_file" ]; then
        debug "load_env_file: file not found: $env_file"
        return 1
    fi
    
    # Read line by line, only export valid KEY=VALUE pairs
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        case "$line" in
            ''|\#*) continue ;;
        esac
        
        # Allow optional "export " prefix
        if [[ "$line" == export\ * ]]; then
            line="${line#export }"
        fi
        
        # Only process lines that look like KEY=VALUE
        # Key must be alphanumeric with underscores, starting with a letter
        if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            # Extract key and value
            local key="${line%%=*}"
            local value="${line#*=}"
            
            # Trim leading/trailing whitespace for unquoted values
            # Keep quoted values as-is (strip surrounding quotes)
            case "$value" in
                \"*\") value="${value:1:-1}" ;;
                \'*\') value="${value:1:-1}" ;;
                *) value="$(printf '%s' "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')" ;;
            esac
            
            # Export the variable
            export "$key=$value"
            debug "load_env_file: exported $key"
        else
            debug "load_env_file: skipping malformed line"
        fi
    done < "$env_file"
    
    return 0
}

# Acquire install lock to avoid concurrent installs
acquire_lock() {
    local lock_path="$1"
    
    if [ -z "$lock_path" ]; then
        err "acquire_lock: empty lock path"
        exit 1
    fi
    
    # Ensure parent directory exists before attempting lock
    local lock_parent
    lock_parent="$(dirname "$lock_path")"
    if [ ! -d "$lock_parent" ]; then
        mkdir -p "$lock_parent" 2>/dev/null || true
    fi
    
    if mkdir "$lock_path" 2>/dev/null; then
        LOCK_DIR="$lock_path"
        LOCK_ACQUIRED=true
        echo "$$" > "$lock_path/pid" 2>/dev/null || true
        debug "Lock acquired: $LOCK_DIR"
        return 0
    fi
    
    # Lock exists - check for stale lock
    if [ -f "$lock_path/pid" ]; then
        local old_pid
        old_pid=$(cat "$lock_path/pid" 2>/dev/null)
        if [ -n "$old_pid" ] && ! kill -0 "$old_pid" 2>/dev/null; then
            warn "Stale lock detected, removing"
            rm -rf "$lock_path" 2>/dev/null || true
            if mkdir "$lock_path" 2>/dev/null; then
                LOCK_DIR="$lock_path"
                LOCK_ACQUIRED=true
                echo "$$" > "$lock_path/pid" 2>/dev/null || true
                debug "Lock acquired after cleanup: $LOCK_DIR"
                return 0
            fi
        fi
    else
        # Lock dir exists but no pid file - stale lock from crash
        warn "Stale lock detected (no pid file), removing"
        rm -rf "$lock_path" 2>/dev/null || true
        if mkdir "$lock_path" 2>/dev/null; then
            LOCK_DIR="$lock_path"
            LOCK_ACQUIRED=true
            echo "$$" > "$lock_path/pid" 2>/dev/null || true
            debug "Lock acquired after cleanup: $LOCK_DIR"
            return 0
        fi
    fi
    
    err "Another installation is already running"
    info "If you're sure it's stale, remove: $lock_path"
    exit 1
}

detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "macos";;
        Linux*)  echo "linux";;
        MINGW*|MSYS*|CYGWIN*) echo "windows";;
        FreeBSD*) echo "freebsd";;
        *)       echo "unknown";;
    esac
}

detect_arch() {
    local arch
    arch=$(uname -m)
    case "$arch" in
        x86_64|amd64) echo "x86_64";;
        aarch64|arm64) echo "arm64";;
        armv7l|armv6l) echo "arm";;
        i386|i686) echo "x86";;
        *) echo "$arch";;
    esac
}

detect_linux_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif [ -f /etc/debian_version ]; then
        echo "debian"
    elif [ -f /etc/redhat-release ]; then
        echo "rhel"
    elif [ -f /etc/alpine-release ]; then
        echo "alpine"
    else
        echo "unknown"
    fi
}

# Check if we have network connectivity
check_network() {
    debug "Checking network connectivity..."
    if [ "$SKIP_NET_CHECK" = "1" ]; then
        warn "Skipping network check (TAU_SKIP_NET_CHECK=1)"
        return 0
    fi
    # Try multiple endpoints in case one is blocked
    # Use retry for flaky connections
    if curl -sf --connect-timeout 5 --max-time 10 --retry 2 --retry-delay 1 "https://github.com" >/dev/null 2>&1; then
        return 0
    elif curl -sf --connect-timeout 5 --max-time 10 --retry 2 --retry-delay 1 "https://astral.sh" >/dev/null 2>&1; then
        return 0
    fi
    
    # Fallback to ping with OS-specific timeout flags
    # macOS: -t is timeout in seconds, no -W equivalent for our use
    # Linux: -W is timeout in seconds
    if [ "$(uname -s)" = "Darwin" ]; then
        if ping -c 1 -t 5 8.8.8.8 >/dev/null 2>&1; then
            return 0
        fi
    else
        if ping -c 1 -W 5 8.8.8.8 >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Check available disk space (in MB)
check_disk_space() {
    local path="$1"
    local required_mb="${2:-500}"  # Default 500MB
    local available_mb
    
    # Ensure parent directory exists for the check
    local check_path="$path"
    while [ ! -d "$check_path" ] && [ "$check_path" != "/" ]; do
        check_path=$(dirname "$check_path")
    done
    
    # df -m works the same on macOS and Linux for our purposes
    # Column 4 is "Available" on both platforms
    available_mb=$(df -m "$check_path" 2>/dev/null | tail -1 | awk '{print $4}')
    
    if [ -n "$available_mb" ] && [ "$available_mb" -lt "$required_mb" ] 2>/dev/null; then
        return 1
    fi
    return 0
}

# Check write permissions for install directory
check_write_permission() {
    local path="$1"
    local check_path="$path"
    
    # Find the first existing parent directory
    while [ ! -d "$check_path" ] && [ "$check_path" != "/" ]; do
        check_path=$(dirname "$check_path")
    done
    
    if [ -w "$check_path" ]; then
        return 0
    fi
    return 1
}

# Check Python version meets minimum requirement
check_python_version() {
    local min_version="${1:-3.8}"
    local python_cmd="${2:-python3}"
    
    if ! command_exists "$python_cmd"; then
        return 1
    fi
    
    local version
    version=$("$python_cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    
    if [ -z "$version" ]; then
        return 1
    fi
    
    # Compare versions (works for X.Y format)
    local min_major min_minor cur_major cur_minor
    min_major=$(echo "$min_version" | cut -d. -f1)
    min_minor=$(echo "$min_version" | cut -d. -f2)
    cur_major=$(echo "$version" | cut -d. -f1)
    cur_minor=$(echo "$version" | cut -d. -f2)
    
    if [ "$cur_major" -gt "$min_major" ]; then
        return 0
    elif [ "$cur_major" -eq "$min_major" ] && [ "$cur_minor" -ge "$min_minor" ]; then
        return 0
    fi
    return 1
}

# Check if user has sudo access (without prompting)
check_sudo_access() {
    # Check if we're already root
    if [ "$(id -u)" = "0" ]; then
        return 0
    fi
    # Check if sudo is available and user can use it
    if command_exists sudo; then
        # This checks if user has NOPASSWD sudo or has recently authenticated
        if sudo -n true 2>/dev/null; then
            return 0
        fi
        # Check if user is in sudo/wheel group
        if groups 2>/dev/null | grep -qE '\b(sudo|wheel|admin)\b'; then
            return 0
        fi
    fi
    return 1
}

# Initialize/detect Homebrew on macOS (both Intel and Apple Silicon)
init_homebrew() {
    if command_exists brew; then
        return 0
    fi
    
    # Check Apple Silicon path first (more common now)
    if [ -f "/opt/homebrew/bin/brew" ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        return 0
    fi
    
    # Check Intel Mac path
    if [ -f "/usr/local/bin/brew" ]; then
        eval "$(/usr/local/bin/brew shellenv)"
        return 0
    fi
    
    return 1
}

generate_instance_id() {
    local path="$1"
    local base=$(basename "$path" | tr '[:upper:]' '[:lower:]' | \
                 sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//;s/-$//')
    
    if [ -z "$base" ] || [ "$base" = "tau" ]; then
        local hash=""
        # Try various hash commands in order of preference
        if command_exists md5; then
            hash=$(echo -n "$path" | md5)
        elif command_exists md5sum; then
            hash=$(echo -n "$path" | md5sum | cut -d' ' -f1)
        elif command_exists shasum; then
            hash=$(echo -n "$path" | shasum | cut -d' ' -f1)
        elif command_exists sha256sum; then
            hash=$(echo -n "$path" | sha256sum | cut -d' ' -f1)
        else
            # Fallback: use a simple checksum based on string length and characters
            hash=$(printf '%s' "$path" | cksum | cut -d' ' -f1)
        fi
        base="${base:-tau}-${hash:0:6}"
    fi
    echo "$base"
}

resolve_bot_name() {
    local token="$1"
    local response
    response=$(curl -sf --connect-timeout 10 --max-time 15 \
        "https://api.telegram.org/bot${token}/getMe" 2>/dev/null) || return 1
    
    # Extract username from JSON: {"ok":true,"result":{..."username":"bot_name"...}}
    local username
    username=$(printf '%s' "$response" | grep -o '"username":"[^"]*"' | head -1 | cut -d'"' -f4)
    
    if [ -n "$username" ]; then
        printf '%s' "$username" | tr '[:upper:]' '[:lower:]'
        return 0
    fi
    return 1
}

collect_bot_token() {
    # Already have a token
    if [ -n "$BOT_TOKEN" ]; then
        return 0
    fi
    
    # Check environment variable
    if [ -n "$TAU_BOT_TOKEN" ]; then
        BOT_TOKEN="$TAU_BOT_TOKEN"
        info "Using provided token"
        return 0
    fi
    
    # Interactive prompt
    if [ "$HAS_TTY" = true ] && [ "$FAST" != "1" ]; then
        info "Create a bot via @BotFather ${SYM_ARROW} /newbot"
        log ""
        printf "  ${DIM}Bot token:${NC} "
        read -r BOT_TOKEN </dev/tty 2>/dev/null || BOT_TOKEN=""
        
        if [ -z "$BOT_TOKEN" ]; then
            err "Token required"
            exit 1
        fi
    elif [ "$FAST" = "1" ]; then
        err "Telegram bot token required in non-interactive mode"
        info "Set TAU_BOT_TOKEN environment variable"
        exit 1
    else
        err "Telegram bot token required"
        info "Set TAU_BOT_TOKEN environment variable and run again"
        exit 1
    fi
    
    # Validate token format (basic check)
    if [[ ! "$BOT_TOKEN" =~ ^[0-9]+:[A-Za-z0-9_-]+$ ]]; then
        warn "Token format looks unusual (expected: 123456:ABC-DEF...)"
        if [ "$HAS_TTY" = true ]; then
            if ! prompt_yn "Continue anyway"; then
                exit 1
            fi
        fi
    fi
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
    printf "   ___________               _______  .__            __        \n"
    printf "   \\__    ___/____   __ __   \\      \\ |__| ____     |__|____   \n"
    printf "     |    |  \\__  \\ |  |  \\  /   |   \\|  |/    \\    |  \\__  \\  \n"
    printf "     |    |   / __ \\|  |  / /    |    \\  |   |  \\   |  |/ __ \\_\n"
    printf "     |____|  (____  /____/  \\____|__  /__|___|  /\\__|  (____  /\n"
    printf "                  \\/                \\/        \\/\\______|    \\/ \n"
    printf "%s\n" "$NC"
    
    local OS=$(detect_os)
    local ARCH=$(detect_arch)
    debug "OS: $OS, Arch: $ARCH, TTY: $HAS_TTY, Color: $HAS_COLOR, Unicode: $HAS_UNICODE"
    
    # Warn about unsupported platforms
    if [ "$OS" = "windows" ]; then
        warn "Windows detected - use WSL2 for best results"
        info "Install WSL2: wsl --install"
    elif [ "$OS" = "unknown" ]; then
        warn "Unknown operating system: $(uname -s)"
        info "Proceeding, but some features may not work"
    fi
    
    # Always collect bot token first to determine install directory
    section "Telegram"
    log ""
    collect_bot_token
    
    # Resolve bot name from token for directory name
    if [ -n "$BOT_TOKEN" ]; then
        BOT_NAME=$(resolve_bot_name "$BOT_TOKEN" 2>/dev/null) || true
        if [ -n "$BOT_NAME" ]; then
            ok "Bot: @$BOT_NAME"
        else
            warn "Could not resolve bot name from token"
        fi
    fi
    
    local default_dir="${BOT_NAME:-tau}"
    local new_dir="$HOME/$default_dir"
    
    # Update install dir and running-from-repo flag
    if [ "$RUNNING_FROM_REPO" = true ] && [ "$new_dir" != "$INSTALL_DIR" ]; then
        # Target differs from repo dir — will clone fresh
        RUNNING_FROM_REPO=false
    fi
    INSTALL_DIR="$new_dir"
    printf "\n"
    
    INSTANCE_ID=$(generate_instance_id "$INSTALL_DIR")
    
    if [ "$RUNNING_FROM_REPO" = true ]; then
        ok "Using existing repo"
    fi
    
    info "Installing to $INSTALL_DIR"
}

# Pre-flight checks before starting installation
step_preflight() {
    section "Pre-flight checks"
    log ""
    
    local errors=0
    local OS=$(detect_os)
    local ARCH=$(detect_arch)
    
    # Check network connectivity
    if check_network; then
        check_ok "Network" "connected"
    else
        check_missing "Network connectivity"
        err "Cannot reach GitHub or Astral.sh"
        info "Check your internet connection and try again"
        errors=$((errors + 1))
    fi
    
    # Check disk space (need ~500MB)
    if check_disk_space "$INSTALL_DIR" 500; then
        check_ok "Disk space" "sufficient"
    else
        check_missing "Disk space"
        err "Need at least 500MB free space"
        errors=$((errors + 1))
    fi
    
    # Check write permissions
    if check_write_permission "$INSTALL_DIR"; then
        check_ok "Write permissions"
    else
        check_missing "Write permissions for $INSTALL_DIR"
        info "Try a different install path or fix permissions"
        errors=$((errors + 1))
    fi
    
    # Architecture info
    debug "Architecture: $ARCH"
    if [ "$ARCH" = "arm" ] || [ "$ARCH" = "x86" ]; then
        warn "32-bit architecture detected - some packages may not be available"
    fi
    
    if [ $errors -gt 0 ]; then
        log ""
        err "Pre-flight checks failed ($errors errors)"
        exit 1
    fi
}

step_dependencies() {
    section "Dependencies"
    log ""
    
    local OS=$(detect_os)
    local ARCH=$(detect_arch)
    local missing_git=false
    local missing_python=false
    local missing_curl=false
    local python_too_old=false
    
    # Check curl
    if command_exists curl; then
        check_ok "curl"
    else
        check_missing "curl"
        missing_curl=true
    fi
    
    # Check git
    if command_exists git; then
        check_ok "git" "$(git --version 2>&1 | cut -d' ' -f3)"
    else
        check_missing "git"
        missing_git=true
    fi
    
    # Check Python with version validation
    if command_exists python3; then
        local py_version
        py_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        if check_python_version "3.8"; then
            check_ok "python3" "$py_version"
        else
            check_missing "python3 >= 3.8 (found $py_version)"
            python_too_old=true
            missing_python=true
        fi
    else
        check_missing "python3"
        missing_python=true
    fi
    
    # Check uv
    if command_exists uv; then
        check_ok "uv" "$(uv --version 2>&1 | head -1 | awk '{print $2}')"
    else
        check_pending "uv"
    fi
    
    # Check cursor agent
    if command_exists agent; then
        check_ok "cursor agent"
    else
        check_pending "cursor agent"
    fi
    
    # Handle missing dependencies
    if [ "$missing_git" = true ] || [ "$missing_python" = true ] || \
       [ "$missing_curl" = true ]; then
        log ""
        warn "Missing or incompatible dependencies"
        
        if [ "$OS" = "macos" ]; then
            # Initialize Homebrew if available
            init_homebrew
            
            if prompt_yn "Install missing"; then
                install_macos_deps "$missing_git" "$missing_curl" "$missing_python"
            fi
        elif [ "$OS" = "linux" ]; then
            if prompt_yn "Install missing"; then
                if ! install_linux_deps; then
                    warn "Automatic installation failed"
                fi
            fi
        elif [ "$OS" = "freebsd" ]; then
            warn "FreeBSD detected"
            info "Install manually: pkg install git python3 curl"
        else
            warn "Automatic installation not available for this OS"
            info "Please install: git, python3 (>=3.8), curl"
        fi
        
        # Re-check after installation attempt
        log ""
        local still_missing=""
        if ! command_exists git; then
            still_missing="${still_missing}git "
        fi
        if ! command_exists python3; then
            still_missing="${still_missing}python3 "
        elif ! check_python_version "3.8"; then
            still_missing="${still_missing}python3>=3.8 "
        fi
        if ! command_exists curl; then
            still_missing="${still_missing}curl "
        fi
        
        if [ -n "$still_missing" ]; then
            err "Still missing: $still_missing"
            info "Install these manually and run the installer again"
            exit 1
        fi
        
        ok "All dependencies satisfied"
    fi
}

install_macos_deps() {
    local need_git="$1"
    local need_curl="$2"
    local need_python="$3"
    
    if [ "$need_git" = true ] || [ "$need_curl" = true ]; then
        if ! xcode-select -p &>/dev/null; then
            info "Installing Xcode Command Line Tools..."
            
            # Start the install
            xcode-select --install 2>/dev/null || true
            
            if [ "$HAS_TTY" = true ]; then
                info "A dialog will appear. Click 'Install' to continue."
                info "Waiting for installation to complete..."
                log ""
                
                # Wait for xcode-select to complete (polls for the directory)
                local max_wait=600  # 10 minutes max
                local waited=0
                while ! xcode-select -p &>/dev/null && [ $waited -lt $max_wait ]; do
                    portable_sleep 5
                    waited=$((waited + 5))
                    # Show progress every 30 seconds
                    if [ $((waited % 30)) -eq 0 ]; then
                        info "Still waiting... (${waited}s)"
                    fi
                done
                
                if xcode-select -p &>/dev/null; then
                    ok "Xcode Command Line Tools installed"
                else
                    warn "Xcode CLT installation may not have completed"
                    info "You can continue, but git/curl may not work"
                    pause
                fi
            else
                # Non-interactive: just warn and continue
                warn "Xcode CLT install started - may need manual completion"
                info "Run: xcode-select --install"
            fi
        fi
    fi
    
    if [ "$need_python" = true ]; then
        # Try to initialize existing Homebrew first
        init_homebrew
        
        if ! command_exists brew; then
            if prompt_yn "Install Homebrew"; then
                info "Installing Homebrew..."
                # Run Homebrew installer
                if [ "$HAS_TTY" = true ]; then
                    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                else
                    # Non-interactive Homebrew install
                    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" </dev/null
                fi
                
                # Initialize brew in current session
                init_homebrew
            fi
        fi
        
        if command_exists brew; then
            run_spin "Installing Python" brew install python3
        else
            warn "Homebrew not available, cannot install Python"
            info "Install Python manually: https://www.python.org/downloads/"
        fi
    fi
}

install_linux_deps() {
    local distro=$(detect_linux_distro)
    local arch=$(detect_arch)
    
    debug "Linux distro: $distro, arch: $arch"
    
    # Check if we need sudo
    local need_sudo=true
    if [ "$(id -u)" = "0" ]; then
        need_sudo=false
    fi
    
    if [ "$need_sudo" = true ] && ! check_sudo_access; then
        warn "sudo access required to install dependencies"
        info "Please run as root or ensure sudo is configured"
        info "Install manually: git python3 python3-pip python3-venv curl"
        return 1
    fi
    
    local sudo_cmd=""
    [ "$need_sudo" = true ] && sudo_cmd="sudo"
    
    case "$distro" in
        ubuntu|debian|pop|linuxmint|raspbian)
            info "Updating package lists..."
            $sudo_cmd apt-get update >/dev/null 2>&1 || {
                warn "apt update failed, continuing anyway..."
            }
            $sudo_cmd apt-get install -y git python3 python3-pip python3-venv curl build-essential || {
                err "Failed to install packages"
                return 1
            }
            ;;
        fedora)
            $sudo_cmd dnf install -y git python3 python3-pip python3-devel curl gcc gcc-c++ make || {
                err "Failed to install packages"
                return 1
            }
            ;;
        rhel|centos|rocky|almalinux|ol)
            # RHEL-based distros may need EPEL
            $sudo_cmd dnf install -y epel-release 2>/dev/null || true
            $sudo_cmd dnf install -y git python3 python3-pip python3-devel curl gcc gcc-c++ make || {
                err "Failed to install packages"
                return 1
            }
            ;;
        arch|manjaro|endeavouros)
            $sudo_cmd pacman -Sy --noconfirm git python python-pip curl base-devel || {
                err "Failed to install packages"
                return 1
            }
            ;;
        alpine)
            $sudo_cmd apk add --no-cache git python3 py3-pip curl build-base || {
                err "Failed to install packages"
                return 1
            }
            ;;
        opensuse*|sles)
            $sudo_cmd zypper install -y git python3 python3-pip curl gcc gcc-c++ make || {
                err "Failed to install packages"
                return 1
            }
            ;;
        void)
            $sudo_cmd xbps-install -Sy git python3 python3-pip curl base-devel || {
                err "Failed to install packages"
                return 1
            }
            ;;
        nixos)
            warn "NixOS detected - use nix-shell or home-manager"
            info "Example: nix-shell -p git python3 curl"
            return 1
            ;;
        *)
            warn "Unknown distro: $distro"
            info "Install manually: git python3 python3-pip python3-venv curl"
            info "Common package managers:"
            info "  apt: sudo apt install git python3 python3-pip python3-venv curl"
            info "  dnf: sudo dnf install git python3 python3-pip curl"
            info "  pacman: sudo pacman -S git python python-pip curl"
            return 1
            ;;
    esac
    
    ok "System packages installed"
}

step_install_uv() {
    # Update PATH to include common uv locations
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    if command_exists uv; then
        debug "uv already installed: $(command -v uv)"
        return 0
    fi
    
    debug "Installing uv..."
    
    local tmplog
    tmplog=$(mktemp)
    CLEANUP_FILES+=("$tmplog")
    
    # Download and run uv installer (with retry for flaky connections)
    if curl -LsSf --retry 3 --retry-delay 2 https://astral.sh/uv/install.sh -o "$tmplog.sh" 2>"$tmplog"; then
        CLEANUP_FILES+=("$tmplog.sh")
        chmod +x "$tmplog.sh"
        
        # Run installer in background and track with spinner
        sh "$tmplog.sh" >"$tmplog" 2>&1 &
        local installer_pid=$!
        
        if ! spinner $installer_pid "Installing uv"; then
            err "uv installer failed"
            if [ -s "$tmplog" ]; then
                info "Error output:"
                tail -10 "$tmplog" | while IFS= read -r line; do
                    info "  $line"
                done
            fi
            rm -f "$tmplog" "$tmplog.sh"
            exit 1
        fi
    else
        err "Failed to download uv installer"
        if [ -s "$tmplog" ]; then
            info "Error: $(cat "$tmplog")"
        fi
        rm -f "$tmplog" "$tmplog.sh"
        exit 1
    fi
    
    rm -f "$tmplog" "$tmplog.sh"
    
    # Update PATH again after install
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    # Source shell config if available (for immediate use)
    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
        if [ -f "$rc" ]; then
            # shellcheck disable=SC1090
            source "$rc" 2>/dev/null || true
            break
        fi
    done
    
    debug "Updated PATH: $PATH"
    
    if ! command_exists uv; then
        err "uv installed but not found in PATH"
        info "uv may have been installed to a non-standard location"
        info "Try: export PATH=\"\$HOME/.local/bin:\$PATH\""
        info "Then run the installer again"
        exit 1
    fi
    
    debug "uv installed at: $(command -v uv)"
    ok "uv installed"
}

step_clone() {
    section "Install"
    log ""
    
    step_install_uv
    
    if [ "$RUNNING_FROM_REPO" = true ]; then
        if prompt_yn "Pull latest"; then
            cd "$INSTALL_DIR" || {
                err "Cannot access $INSTALL_DIR"
                exit 1
            }
            git pull >/dev/null 2>&1 &
            spinner $! "Pulling" true || warn "Could not pull (continuing with existing code)"
        fi
        
        cd "$INSTALL_DIR" || {
            err "Cannot access $INSTALL_DIR"
            exit 1
        }
        return 0
    fi
    
    if [ -d "$INSTALL_DIR" ]; then
        # Check if directory is effectively empty (only .install.lock from acquire_lock)
        local dir_contents
        dir_contents=$(ls -A "$INSTALL_DIR" 2>/dev/null | grep -v '^\.install\.lock$' || true)
        
        if [ -z "$dir_contents" ]; then
            # Directory only has our lock (or is empty) — remove so git clone can use it
            debug "Directory exists but only contains install lock, removing for fresh clone"
            rm -rf "$INSTALL_DIR"
        elif [ ! -f "$INSTALL_DIR/tauctl" ]; then
            # Directory exists with real content but isn't a tau install
            warn "Directory exists but appears incomplete"
            if prompt_yn "Remove and re-clone"; then
                if ! safe_rm_dir "$INSTALL_DIR"; then
                    err "Failed to remove directory safely"
                    exit 1
                fi
            else
                err "Cannot continue with incomplete installation"
                info "Remove $INSTALL_DIR manually and run the installer again"
                exit 1
            fi
        else
            # Existing tau installation
            if prompt_yn "Pull latest"; then
                cd "$INSTALL_DIR" || exit 1
                git pull >/dev/null 2>&1 &
                spinner $! "Pulling" true || warn "Could not pull"
            fi
            cd "$INSTALL_DIR" || exit 1
            return 0
        fi
    fi
    
    # Fresh clone
    local parent_dir
    parent_dir=$(dirname "$INSTALL_DIR")
    
    if [ ! -d "$parent_dir" ]; then
        debug "Creating parent directory: $parent_dir"
        if ! mkdir -p "$parent_dir"; then
            err "Cannot create directory: $parent_dir"
            exit 1
        fi
    fi
    
    local tmplog
    tmplog=$(mktemp)
    CLEANUP_FILES+=("$tmplog")
    
    # Clone with timeout protection
    # GIT_HTTP_LOW_SPEED_LIMIT: minimum bytes/sec before considering stalled
    # GIT_HTTP_LOW_SPEED_TIME: seconds to wait at low speed before aborting
    (
        export GIT_HTTP_LOW_SPEED_LIMIT=1000
        export GIT_HTTP_LOW_SPEED_TIME=60
        git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
    ) >"$tmplog" 2>&1 &
    
    if ! spinner $! "Cloning repository"; then
        err "Failed to clone repository"
        if [ -s "$tmplog" ]; then
            info "Git error:"
            tail -5 "$tmplog" | while IFS= read -r line; do
                info "  $line"
            done
        fi
        rm -f "$tmplog"
        exit 1
    fi
    rm -f "$tmplog"
    
    # Re-acquire lock inside cloned directory (lock was removed for clean clone)
    if [ "$LOCK_ACQUIRED" = false ] || [ ! -d "$INSTALL_DIR/.install.lock" ]; then
        LOCK_ACQUIRED=false
        acquire_lock "$INSTALL_DIR/.install.lock"
    fi
    
    cd "$INSTALL_DIR" || {
        err "Clone succeeded but cannot access $INSTALL_DIR"
        exit 1
    }
}

step_python() {
    cd "$INSTALL_DIR" || { err "Cannot access $INSTALL_DIR"; exit 1; }
    debug "INSTALL_DIR=$INSTALL_DIR"
    debug "PATH=$PATH"
    
    # Ensure uv is available (might need PATH update after install)
    if ! command_exists uv; then
        debug "uv not found, updating PATH"
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        if ! command_exists uv; then
            err "uv not found in PATH"
            info "Try: export PATH=\"\$HOME/.local/bin:\$PATH\""
            exit 1
        fi
    fi
    debug "uv location: $(command -v uv)"
    debug "uv version: $(uv --version 2>&1 || echo 'unknown')"
    debug "python3 location: $(command -v python3 2>/dev/null || echo 'not found')"
    debug "python3 version: $(python3 --version 2>&1 || echo 'unknown')"
    
    if [ ! -d ".venv" ]; then
        debug "Creating venv in $(pwd)/.venv"
        # Create venv with explicit error handling
        if ! uv venv .venv >/dev/null 2>&1; then
            err "Failed to create virtual environment"
            info "Attempting with verbose output..."
            log ""
            uv venv .venv 2>&1 | head -20 || true
            log ""
            err "uv venv failed - check Python installation"
            info "Try: python3 --version"
            info "Try: uv --version"
            info "Try: TAU_DEBUG=1 ./install.sh"
            exit 1
        fi
        ok "Created environment"
    else
        debug ".venv already exists"
    fi
    
    if [ ! -f ".venv/bin/activate" ]; then
        err "Virtual environment missing activate script"
        debug "Contents of .venv: $(ls -la .venv 2>&1 || echo 'cannot list')"
        exit 1
    fi
    
    debug "Activating venv"
    source .venv/bin/activate
    debug "Activated, python now: $(command -v python)"
    
    run_silent "Installing tau" uv pip install -e .
    run_silent "Installing supervisor" uv pip install supervisor
    
    mkdir -p "$INSTALL_DIR/context/logs"
}

step_cursor_agent() {
    # Update PATH to include Cursor locations
    export PATH="$HOME/.cursor/bin:$HOME/.local/bin:$PATH"
    
    if command_exists agent; then
        debug "Cursor agent already installed: $(command -v agent)"
        # Check if already authenticated
        if agent whoami >/dev/null 2>&1; then
            debug "Cursor agent already authenticated"
            return 0
        fi
        # Installed but not authenticated - prompt login
        if [ "$HAS_TTY" = true ]; then
            warn "Cursor agent not authenticated"
            log ""
            if agent login; then
                ok "Authenticated"
            else
                warn "Authentication failed"
                info "Run 'agent login' manually to authenticate"
            fi
        else
            info "Run 'agent login' to authenticate"
        fi
        return 0
    fi
    
    log ""
    info "Cursor agent CLI enables autonomous operation"
    
    if prompt_yn "Install Cursor CLI"; then
        local tmplog
        tmplog=$(mktemp)
        CLEANUP_FILES+=("$tmplog")
        
        info "Downloading Cursor CLI..."
        if curl -fsSL --retry 3 --retry-delay 2 https://cursor.com/install -o "$tmplog.sh" 2>"$tmplog"; then
            CLEANUP_FILES+=("$tmplog.sh")
            chmod +x "$tmplog.sh"
            
            if bash "$tmplog.sh" >"$tmplog" 2>&1; then
                export PATH="$HOME/.cursor/bin:$HOME/.local/bin:$PATH"
                
                if command_exists agent; then
                    ok "Cursor CLI installed"
                    
                    # Only attempt login if we have a TTY
                    if [ "$HAS_TTY" = true ]; then
                        info "Opening browser for authentication..."
                        log ""
                        if agent login 2>/dev/null; then
                            ok "Authenticated"
                        else
                            warn "Authentication may have failed"
                            info "Run 'agent login' manually to authenticate"
                        fi
                    else
                        info "Run 'agent login' to authenticate after installation"
                    fi
                else
                    warn "Installed but not found in PATH"
                    info "Restart terminal, then run: agent login"
                fi
            else
                warn "Cursor CLI installation failed"
                if [ -s "$tmplog" ]; then
                    debug "Error: $(tail -5 "$tmplog")"
                fi
            fi
        else
            warn "Failed to download Cursor CLI installer"
        fi
        
        rm -f "$tmplog" "$tmplog.sh"
    else
        info "Tau will work without it (limited functionality)"
        info "Install later: curl https://cursor.com/install -fsSL | bash"
    fi
}

step_telegram() {
    # Token should already be collected in step_welcome.
    # If somehow missing, prompt for it.
    if [ -z "$BOT_TOKEN" ]; then
        section "Telegram"
        log ""
        collect_bot_token
        ok "Telegram configured"
    fi
    
    # Write/update .env file
    if [ -f "$INSTALL_DIR/.env" ]; then
        # Remove existing token line (if any) and append new one
        grep -v "^TAU_BOT_TOKEN=" "$INSTALL_DIR/.env" > "$INSTALL_DIR/.env.tmp" 2>/dev/null || true
        mv "$INSTALL_DIR/.env.tmp" "$INSTALL_DIR/.env"
        echo "TAU_BOT_TOKEN=$BOT_TOKEN" >> "$INSTALL_DIR/.env"
    else
        echo "TAU_BOT_TOKEN=$BOT_TOKEN" > "$INSTALL_DIR/.env"
    fi
}

step_openai() {
    local ENV_FILE="$INSTALL_DIR/.env"
    
    # Check if already set in environment
    if [ -n "$OPENAI_API_KEY" ]; then
        if ! grep -q "OPENAI_API_KEY" "$ENV_FILE" 2>/dev/null; then
            echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> "$ENV_FILE"
            debug "Added OpenAI key from environment"
        fi
        return 0
    fi
    
    # Check if already in .env
    if grep -q "OPENAI_API_KEY=" "$ENV_FILE" 2>/dev/null; then
        debug "OpenAI key already in .env"
        return 0
    fi
    
    # Skip in non-interactive mode
    [ "$FAST" = "1" ] && return 0
    [ "$HAS_TTY" = false ] && return 0
    
    log ""
    info "OpenAI API key enables voice features (optional)"
    printf "  ${DIM}OpenAI key (Enter to skip):${NC} "
    read -r API_KEY </dev/tty 2>/dev/null || API_KEY=""
    
    if [ -n "$API_KEY" ]; then
        # Basic validation
        if [[ "$API_KEY" =~ ^sk- ]]; then
            echo "OPENAI_API_KEY=$API_KEY" >> "$ENV_FILE"
            ok "OpenAI key configured"
        else
            warn "Key doesn't look like an OpenAI key (should start with 'sk-')"
            if prompt_yn "Save anyway"; then
                echo "OPENAI_API_KEY=$API_KEY" >> "$ENV_FILE"
            fi
        fi
    else
        info "Skipped - add OPENAI_API_KEY to .env later for voice"
    fi
}

setup_launchd() {
    local PLIST_DIR="$HOME/Library/LaunchAgents"
    local PLIST_NAME="com.tau.$INSTANCE_ID.supervisor"
    local PLIST_FILE="$PLIST_DIR/$PLIST_NAME.plist"
    local GUI_UID
    GUI_UID=$(id -u)
    
    mkdir -p "$PLIST_DIR"
    
    # Ensure log directory exists
    mkdir -p "$INSTALL_DIR/context/logs"
    
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
        <string>$INSTALL_DIR/.venv/bin:$HOME/.local/bin:$HOME/.cursor/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
EOF
    
    # Unload if already loaded (ignore errors)
    launchctl bootout "gui/$GUI_UID/$PLIST_NAME" 2>/dev/null || \
        launchctl unload "$PLIST_FILE" 2>/dev/null || true
    
    # Small delay to ensure unload completes
    portable_sleep 1
    
    # Load using modern bootstrap (preferred) or fallback to load
    if ! launchctl bootstrap "gui/$GUI_UID" "$PLIST_FILE" 2>/dev/null; then
        # Fallback for older macOS versions
        if ! launchctl load "$PLIST_FILE" 2>/dev/null; then
            err "Failed to load launchd service"
            info "Try manually: launchctl load $PLIST_FILE"
            return 1
        fi
    fi
    
    ok "Auto-start configured (launchd)"
}

systemd_user_available() {
    # Check if systemd user services are available
    # This fails in containers without D-Bus or systemd
    systemctl --user status >/dev/null 2>&1
}

setup_systemd() {
    local SERVICE_DIR="$HOME/.config/systemd/user"
    local SERVICE_NAME="tau-$INSTANCE_ID"
    local SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME.service"
    
    if ! systemd_user_available; then
        warn "systemd user services not available (container?)"
        info "Skipping auto-start, will start manually"
        return 1
    fi
    
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
    
    systemctl --user daemon-reload || { warn "Failed to reload systemd"; return 1; }
    systemctl --user enable "$SERVICE_NAME" >/dev/null 2>&1 || true
    if ! systemctl --user start "$SERVICE_NAME"; then
        err "Failed to start systemd service: $SERVICE_NAME"
        info "Check logs: journalctl --user -u $SERVICE_NAME"
        return 1
    fi
    loginctl enable-linger "$USER" 2>/dev/null || true
}

step_launch() {
    section "Launch"
    
    local do_autostart=false
    local do_start=false
    
    # Check if already running (validate PID belongs to supervisord to avoid false positives from PID recycling)
    if [ -f "$INSTALL_DIR/.supervisord.pid" ]; then
        local pid=$(cat "$INSTALL_DIR/.supervisord.pid" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            local proc_name=""
            # Check process name on Linux via /proc
            if [ -f "/proc/$pid/cmdline" ]; then
                proc_name=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
            fi
            # Fallback to ps on macOS/other
            if [ -z "$proc_name" ]; then
                proc_name=$(ps -p "$pid" -o args= 2>/dev/null || true)
            fi
            if echo "$proc_name" | grep -q "supervisord" 2>/dev/null; then
                print_final
                return
            else
                debug "PID $pid exists but is not supervisord (got: $proc_name), ignoring stale pid file"
            fi
        fi
    fi
    
    do_autostart=true
    do_start=true
    
    if [ "$do_autostart" = true ]; then
        local OS=$(detect_os)
        local autostart_ok=false
        
        if [ "$OS" = "macos" ]; then
            setup_launchd
            autostart_ok=true
            # launchd already started supervisord, wait for it
            if [ "$do_start" = true ]; then
                wait_for_ready &
                spinner $! "Starting" true
                print_final
                return
            fi
        elif [ "$OS" = "linux" ]; then
            if setup_systemd; then
                autostart_ok=true
                # systemd already started supervisord, wait for it
                if [ "$do_start" = true ]; then
                    wait_for_ready &
                    spinner $! "Starting" true
                    print_final
                    return
                fi
            fi
            # If systemd setup failed, fall through to manual start
        fi
    fi
    
    if [ "$do_start" = true ]; then
        cd "$INSTALL_DIR" || { err "Cannot access $INSTALL_DIR"; exit 1; }
        
        # Load environment variables safely
        if [ -f .env ]; then
            load_env_file .env || warn "Could not load .env file"
        fi
        
        # Try supervisord first, fall back to direct run for containers
        if "$INSTALL_DIR/tauctl" start >/dev/null 2>&1; then
            print_final
        else
            # Supervisord failed - likely a container, run directly
            # IMPORTANT: Kill any partially-started supervisord and tau processes
            # to avoid two instances polling the same Telegram token (409 conflict)
            if [ -f "$INSTALL_DIR/.supervisord.pid" ]; then
                local spid
                spid=$(cat "$INSTALL_DIR/.supervisord.pid" 2>/dev/null)
                if [ -n "$spid" ] && kill -0 "$spid" 2>/dev/null; then
                    debug "Killing partially-started supervisord (pid $spid)"
                    kill "$spid" 2>/dev/null || true
                    portable_sleep 2
                    kill -9 "$spid" 2>/dev/null || true
                fi
                rm -f "$INSTALL_DIR/.supervisord.pid"
            fi
            rm -f "$INSTALL_DIR/.supervisor.sock"
            # Also kill any orphaned tau processes from the failed supervisord start
            pkill -f "python.*-m tau.*$INSTALL_DIR" 2>/dev/null || true
            portable_sleep 1

            warn "Supervisord failed, starting directly..."
            log ""
            info "Running in foreground (Ctrl+C to stop)"
            info "For background: nohup ./tauctl run &"
            log ""
            exec "$INSTALL_DIR/tauctl" run
        fi
    else
        log ""
        info "Run ./tauctl start when ready"
        info "In Docker: ./tauctl run"
        log ""
    fi
}

wait_for_ready() {
    local retries=30
    local waited=0
    
    while [ $retries -gt 0 ]; do
        if [ -f "$INSTALL_DIR/.supervisord.pid" ]; then
            local pid
            pid=$(cat "$INSTALL_DIR/.supervisord.pid" 2>/dev/null)
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                # Supervisord is running, check if tau is up
                if [ -S "$INSTALL_DIR/.supervisor.sock" ]; then
                    portable_sleep 1
                    return 0
                fi
            fi
        fi
        portable_sleep 0.5
        retries=$((retries - 1))
        waited=$((waited + 1))
        debug "Waiting for supervisord... ($waited)"
    done
    
    warn "Supervisord did not start within expected time"
    return 1
}

print_final() {
    log ""
    if [ "$HAS_UNICODE" = true ]; then
        printf "🥷 ${GREEN}Tau Ninja is live${NC}\n"
    else
        printf "${GREEN}${SYM_OK}${NC} ${GREEN}Tau Ninja is live${NC}\n"
    fi
    log ""
    printf "  ${BOLD}Message your bot to begin${NC}\n"
    log ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

detect_install_context

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --debug)
                DEBUG=1
                shift
                ;;
            --fast|--noninteractive|-y)
                FAST=1
                shift
                ;;
            --quiet|-q)
                QUIET=1
                shift
                ;;
            --help|-h)
                printf "Tau Installation Script\n\n"
                printf "Usage: %s [OPTIONS] [INSTALL_PATH]\n\n" "$0"
                printf "Options:\n"
                printf "  --debug          Enable debug output\n"
                printf "  --fast, -y       Non-interactive mode (accept defaults)\n"
                printf "  --quiet, -q      Suppress most output\n"
                printf "  --help, -h       Show this help\n"
                printf "\nEnvironment Variables:\n"
                printf "  TAU_INSTALL_DIR  Installation directory\n"
                printf "  TAU_BOT_TOKEN    Telegram bot token\n"
                printf "  TAU_DEBUG=1      Enable debug output\n"
                printf "  TAU_FAST=1       Non-interactive mode\n"
                printf "  TAU_QUIET=1      Suppress output\n"
                exit 0
                ;;
            -*)
                warn "Unknown option: $1"
                shift
                ;;
            *)
                INSTALL_DIR="${1/#\~/$HOME}"
                if [ -f "$INSTALL_DIR/tauctl" ] && [ -d "$INSTALL_DIR/tau" ]; then
                    RUNNING_FROM_REPO=true
                else
                    RUNNING_FROM_REPO=false
                fi
                shift
                ;;
        esac
    done
    
    step_welcome
    acquire_lock "$INSTALL_DIR/.install.lock"
    step_preflight
    step_dependencies
    step_clone
    step_python
    step_cursor_agent
    step_telegram
    step_openai
    step_launch
}

main "$@"
