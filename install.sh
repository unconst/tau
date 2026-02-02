#!/usr/bin/env bash
#
# Tau Installation Script
# 
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/unconst/tau/main/install.sh | bash
#
# Or clone and run locally:
#   ./install.sh
#

set -e

# Colors for output (using $'...' syntax for proper escape handling)
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'
CYAN=$'\033[0;36m'
NC=$'\033[0m'
BOLD=$'\033[1m'

# Default installation directory
INSTALL_DIR="${TAU_INSTALL_DIR:-$HOME/tau}"
REPO_URL="${TAU_REPO_URL:-https://github.com/unconst/tau.git}"

# Store the token for later use
BOT_TOKEN=""

# Print colored output using printf for reliability
print_header() {
    printf "\n"
    printf "%s%s═══════════════════════════════════════════════════════════%s\n" "$BOLD" "$CYAN" "$NC"
    printf "%s%s  %s%s\n" "$BOLD" "$CYAN" "$1" "$NC"
    printf "%s%s═══════════════════════════════════════════════════════════%s\n" "$BOLD" "$CYAN" "$NC"
    printf "\n"
}

print_step() {
    printf "%s→%s %s\n" "$BLUE" "$NC" "$1"
}

print_success() {
    printf "%s✓%s %s\n" "$GREEN" "$NC" "$1"
}

print_warning() {
    printf "%s⚠%s %s\n" "$YELLOW" "$NC" "$1"
}

print_error() {
    printf "%s✗%s %s\n" "$RED" "$NC" "$1"
}

print_info() {
    printf "%sℹ%s %s\n" "$CYAN" "$NC" "$1"
}

# Wait for user to continue (read from /dev/tty for curl pipe compatibility)
wait_continue() {
    printf "\n"
    printf "Press Enter to continue..."
    read -r </dev/tty
    printf "\n"
}

# Read user input (compatible with curl pipe)
read_input() {
    read -r "$@" </dev/tty
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*)    OS="macos";;
        Linux*)     OS="linux";;
        *)          OS="unknown";;
    esac
    echo "$OS"
}

# Step 1: Welcome
step_welcome() {
    print_header "Welcome to Tau"
    
    printf "%sTau%s is a self-adapting autonomous agent with Telegram integration.\n" "$BOLD" "$NC"
    printf "\n"
    printf "This installer will guide you through:\n"
    printf "\n"
    printf "  %sStep 1%s → Check system dependencies\n" "$CYAN" "$NC"
    printf "  %sStep 2%s → Install uv (Python package manager)\n" "$CYAN" "$NC"
    printf "  %sStep 3%s → Clone the Tau repository\n" "$CYAN" "$NC"
    printf "  %sStep 4%s → Set up Python environment\n" "$CYAN" "$NC"
    printf "  %sStep 5%s → Install Cursor agent CLI\n" "$CYAN" "$NC"
    printf "  %sStep 6%s → Configure Telegram bot token\n" "$CYAN" "$NC"
    printf "  %sStep 7%s → Configure OpenAI API key (optional)\n" "$CYAN" "$NC"
    printf "  %sStep 8%s → Launch Tau!\n" "$CYAN" "$NC"
    printf "\n"
    printf "Installation directory: %s%s%s\n" "$BOLD" "$INSTALL_DIR" "$NC"
    printf "\n"
    
    printf "Ready to begin? (Y/n): "
    read -r -n 1 REPLY </dev/tty
    printf "\n"
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        printf "Installation cancelled.\n"
        exit 0
    fi
}

# Step 2: Check dependencies
step_check_dependencies() {
    print_header "Step 1: Checking Dependencies"
    
    local missing_critical=false
    
    # Check git
    print_step "Checking for git..."
    if command_exists git; then
        print_success "git is installed ($(git --version | cut -d' ' -f3))"
    else
        print_error "git is not installed"
        missing_critical=true
    fi
    
    # Check Python
    print_step "Checking for Python 3..."
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION is installed"
    else
        print_error "Python 3 is not installed"
        missing_critical=true
    fi
    
    # Check uv
    print_step "Checking for uv..."
    if command_exists uv; then
        print_success "uv is installed"
    else
        print_warning "uv is not installed (will install in next step)"
    fi
    
    # Check Cursor agent CLI
    print_step "Checking for Cursor agent CLI..."
    if command_exists agent; then
        print_success "Cursor agent CLI is installed"
    else
        print_warning "Cursor agent CLI is not installed (will guide installation)"
    fi
    
    # Exit if critical dependencies are missing
    if [ "$missing_critical" = true ]; then
        echo ""
        print_error "Missing critical dependencies. Please install git and Python 3 first."
        exit 1
    fi
    
    wait_continue
}

# Step 3: Install uv
step_install_uv() {
    print_header "Step 2: Installing uv"
    
    if command_exists uv; then
        print_success "uv is already installed"
        wait_continue
        return
    fi
    
    echo "uv is a fast Python package manager that Tau uses."
    echo ""
    print_step "Installing uv from https://astral.sh/uv..."
    echo ""
    
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    if command_exists uv; then
        echo ""
        print_success "uv installed successfully!"
    else
        print_error "Failed to install uv"
        exit 1
    fi
    
    wait_continue
}

# Step 4: Clone repository
step_clone_repo() {
    print_header "Step 3: Cloning Tau Repository"
    
    if [ -d "$INSTALL_DIR" ]; then
        print_info "Tau directory already exists at $INSTALL_DIR"
        printf "\n"
        printf "Update existing installation? (y/N): "
        read -r -n 1 REPLY </dev/tty
        printf "\n"
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_step "Updating repository..."
            cd "$INSTALL_DIR"
            git pull origin main || print_warning "Could not pull updates"
        else
            print_info "Using existing installation"
        fi
    else
        print_step "Cloning from $REPO_URL..."
        echo ""
        git clone "$REPO_URL" "$INSTALL_DIR"
        echo ""
        print_success "Repository cloned to $INSTALL_DIR"
    fi
    
    cd "$INSTALL_DIR"
    wait_continue
}

# Step 5: Setup Python environment
step_setup_python() {
    print_header "Step 4: Setting Up Python Environment"
    
    cd "$INSTALL_DIR"
    
    # Create virtual environment
    if [ ! -d ".venv" ]; then
        print_step "Creating Python virtual environment..."
        uv venv .venv
        print_success "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
    
    echo ""
    
    # Activate and install
    print_step "Activating virtual environment..."
    source .venv/bin/activate
    print_success "Virtual environment activated"
    
    echo ""
    
    print_step "Installing Tau and dependencies..."
    echo ""
    uv pip install -e .
    echo ""
    print_success "Dependencies installed!"
    
    wait_continue
}

# Step 6: Setup Cursor agent
step_setup_cursor_agent() {
    print_header "Step 5: Setting Up Cursor Agent CLI"
    
    if command_exists agent; then
        print_success "Cursor agent CLI is already installed!"
        wait_continue
        return
    fi
    
    printf "%sThe Cursor agent CLI is required for Tau to function.%s\n" "$YELLOW" "$NC"
    printf "\n"
    printf "To install it:\n"
    printf "\n"
    printf "  %s1.%s Open %sCursor IDE%s\n" "$BOLD" "$NC" "$CYAN" "$NC"
    printf "\n"
    printf "  %s2.%s Press %sCmd+Shift+P%s (macOS) or %sCtrl+Shift+P%s (Linux)\n" "$BOLD" "$NC" "$CYAN" "$NC" "$CYAN" "$NC"
    printf "\n"
    printf "  %s3.%s Type: %sInstall 'agent' command%s\n" "$BOLD" "$NC" "$CYAN" "$NC"
    printf "\n"
    printf "  %s4.%s Select the option to install the CLI\n" "$BOLD" "$NC"
    printf "\n"
    
    printf "Press Enter once you've installed the Cursor agent CLI (or 's' to skip): "
    read -r -n 1 REPLY </dev/tty
    printf "\n"
    
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        if command_exists agent; then
            print_success "Cursor agent CLI is now installed!"
        else
            print_warning "Cursor agent CLI not detected yet."
            print_info "You can install it later - Tau will still start."
        fi
    else
        print_info "Skipped. You can install it later."
    fi
    
    wait_continue
}

# Step 7: Setup Telegram token
step_setup_telegram_token() {
    print_header "Step 6: Configure Telegram Bot Token"
    
    printf "Tau communicates with you through a Telegram bot.\n"
    printf "\n"
    printf "%sTo create your Telegram bot:%s\n" "$BOLD" "$NC"
    printf "\n"
    printf "  %s1.%s Open Telegram on your phone or desktop\n" "$BOLD" "$NC"
    printf "\n"
    printf "  %s2.%s Search for %s@BotFather%s and start a chat\n" "$BOLD" "$NC" "$CYAN" "$NC"
    printf "\n"
    printf "  %s3.%s Send %s/newbot%s\n" "$BOLD" "$NC" "$CYAN" "$NC"
    printf "\n"
    printf "  %s4.%s Follow the prompts to name your bot\n" "$BOLD" "$NC"
    printf "\n"
    printf "  %s5.%s Copy the %sHTTP API token%s that BotFather gives you\n" "$BOLD" "$NC" "$YELLOW" "$NC"
    printf "     (looks like: %s123456789:ABCdefGHIjklMNOpqrsTUVwxyz%s)\n" "$CYAN" "$NC"
    printf "\n"
    
    # Check if token is already set in environment
    if [ -n "$TAU_BOT_TOKEN" ]; then
        print_info "TAU_BOT_TOKEN is already set in your environment"
        BOT_TOKEN="$TAU_BOT_TOKEN"
        printf "Use existing token? (Y/n): "
        read -r -n 1 REPLY </dev/tty
        printf "\n"
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            BOT_TOKEN=""
        fi
    fi
    
    if [ -z "$BOT_TOKEN" ]; then
        printf "\n"
        printf "Paste your Telegram Bot Token: "
        read -r BOT_TOKEN </dev/tty
        
        if [ -z "$BOT_TOKEN" ]; then
            print_error "No token provided. Tau requires a Telegram bot token to run."
            printf "\n"
            printf "Try again? (Y/n): "
            read -r -n 1 REPLY </dev/tty
            printf "\n"
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                step_setup_telegram_token
                return
            else
                print_error "Cannot continue without a Telegram bot token."
                exit 1
            fi
        fi
    fi
    
    # Save to .env file
    ENV_FILE="$INSTALL_DIR/.env"
    echo "TAU_BOT_TOKEN=$BOT_TOKEN" > "$ENV_FILE"
    
    echo ""
    print_success "Token saved to $ENV_FILE"
    
    wait_continue
}

# Step 8: Setup OpenAI key (optional)
step_setup_openai_key() {
    print_header "Step 7: Configure OpenAI API Key (Optional)"
    
    printf "OpenAI API key enables voice message transcription.\n"
    printf "\n"
    printf "This is %soptional%s - Tau works without it.\n" "$BOLD" "$NC"
    printf "\n"
    
    if [ -n "$OPENAI_API_KEY" ]; then
        print_info "OPENAI_API_KEY is already set in your environment"
        # Append to .env if not already there
        ENV_FILE="$INSTALL_DIR/.env"
        if ! grep -q "OPENAI_API_KEY" "$ENV_FILE" 2>/dev/null; then
            echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> "$ENV_FILE"
        fi
        wait_continue
        return
    fi
    
    printf "Enter OpenAI API Key (or press Enter to skip): "
    read -r API_KEY </dev/tty
    
    if [ -n "$API_KEY" ]; then
        ENV_FILE="$INSTALL_DIR/.env"
        echo "OPENAI_API_KEY=$API_KEY" >> "$ENV_FILE"
        print_success "API key saved"
    else
        print_info "Skipped. Voice features will be disabled."
    fi
    
    wait_continue
}

# Step 9: Launch Tau
step_launch_tau() {
    print_header "Step 8: Launching Tau!"
    
    printf "%sInstallation complete!%s\n" "$GREEN" "$NC"
    printf "\n"
    printf "Tau is ready to run.\n"
    printf "\n"
    printf "%sWhat happens next:%s\n" "$BOLD" "$NC"
    printf "\n"
    printf "  1. Tau will start and connect to Telegram\n"
    printf "  2. Open Telegram and find your bot\n"
    printf "  3. Send %s/start%s to begin\n" "$CYAN" "$NC"
    printf "\n"
    printf "%sAvailable commands:%s\n" "$BOLD" "$NC"
    printf "\n"
    printf "  %s/start%s     - Initialize the bot\n" "$CYAN" "$NC"
    printf "  %s/task%s      - Add a new task\n" "$CYAN" "$NC"
    printf "  %s/status%s    - Check current status\n" "$CYAN" "$NC"
    printf "  %s/adapt%s     - Self-modify the bot\n" "$CYAN" "$NC"
    printf "  %s/restart%s   - Restart the bot\n" "$CYAN" "$NC"
    printf "\n"
    printf "%sTo run Tau later:%s\n" "$BOLD" "$NC"
    printf "\n"
    printf "  %scd %s && ./start.sh%s\n" "$CYAN" "$INSTALL_DIR" "$NC"
    printf "\n"
    
    # Create start.sh for future use
    START_SCRIPT="$INSTALL_DIR/start.sh"
    cat > "$START_SCRIPT" << 'STARTEOF'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Activate virtual environment and run
source .venv/bin/activate
python -m tau
STARTEOF
    chmod +x "$START_SCRIPT"
    
    printf "Launch Tau now? (Y/n): "
    read -r -n 1 REPLY </dev/tty
    printf "\n"
    
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        printf "\n"
        print_step "Starting Tau..."
        echo ""
        
        cd "$INSTALL_DIR"
        
        # Load environment variables
        set -a
        source .env
        set +a
        
        # Activate virtual environment
        source .venv/bin/activate
        
        # Run Tau
        exec python -m tau
    else
        printf "\n"
        print_info "To start Tau later, run:"
        printf "\n"
        printf "  %scd %s && ./start.sh%s\n" "$CYAN" "$INSTALL_DIR" "$NC"
        printf "\n"
    fi
}

# Main installation flow
main() {
    # Handle --systemd flag
    if [[ "$1" == "--systemd" ]]; then
        if [[ "$(detect_os)" != "linux" ]]; then
            print_error "Systemd services are only available on Linux"
            exit 1
        fi
        
        SERVICE_FILE="$HOME/.config/systemd/user/tau.service"
        mkdir -p "$(dirname "$SERVICE_FILE")"
        
        cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Tau Self-Adapting Agent
After=network.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/start.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF
        
        print_success "Created systemd service at $SERVICE_FILE"
        echo ""
        echo "To enable and start:"
        echo "  systemctl --user daemon-reload"
        echo "  systemctl --user enable tau"
        echo "  systemctl --user start tau"
        exit 0
    fi
    
    # Handle custom install directory
    if [[ -n "$1" ]] && [[ "$1" != --* ]]; then
        INSTALL_DIR="$1"
    fi
    
    # Run installation steps
    step_welcome
    step_check_dependencies
    step_install_uv
    step_clone_repo
    step_setup_python
    step_setup_cursor_agent
    step_setup_telegram_token
    step_setup_openai_key
    step_launch_tau
}

# Run main function
main "$@"
