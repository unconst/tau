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

# Detect Linux distro
detect_linux_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif [ -f /etc/debian_version ]; then
        echo "debian"
    elif [ -f /etc/redhat-release ]; then
        echo "rhel"
    else
        echo "unknown"
    fi
}

# Install macOS developer tools (Xcode Command Line Tools)
install_macos_dev_tools() {
    print_step "Installing Xcode Command Line Tools..."
    printf "\n"
    printf "%sThis will open a dialog to install Apple's developer tools.%s\n" "$YELLOW" "$NC"
    printf "These tools include git, compilers, and other essentials.\n"
    printf "\n"
    
    # Check if already installed
    if xcode-select -p &>/dev/null; then
        print_success "Xcode Command Line Tools already installed"
        return 0
    fi
    
    # Trigger the install dialog
    xcode-select --install 2>/dev/null || true
    
    printf "\n"
    printf "%sPlease complete the installation in the dialog that appeared.%s\n" "$YELLOW" "$NC"
    printf "Press Enter once the installation is complete..."
    read -r </dev/tty
    
    # Verify installation
    if xcode-select -p &>/dev/null; then
        print_success "Xcode Command Line Tools installed successfully"
        return 0
    else
        print_warning "Could not verify installation. Continuing anyway..."
        return 1
    fi
}

# Install Linux dependencies based on distro
install_linux_dependencies() {
    local distro=$(detect_linux_distro)
    
    print_step "Installing system dependencies for $distro..."
    printf "\n"
    
    case "$distro" in
        ubuntu|debian|pop|linuxmint|elementary)
            print_info "Using apt package manager"
            printf "\n"
            printf "This will run: %ssudo apt update && sudo apt install -y git python3 python3-pip python3-venv curl build-essential%s\n" "$CYAN" "$NC"
            printf "\n"
            printf "Continue? (Y/n): "
            read -r -n 1 REPLY </dev/tty
            printf "\n"
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                sudo apt update
                sudo apt install -y git python3 python3-pip python3-venv curl build-essential
            fi
            ;;
        fedora|rhel|centos|rocky|alma)
            print_info "Using dnf package manager"
            printf "\n"
            printf "This will run: %ssudo dnf install -y git python3 python3-pip python3-devel curl gcc gcc-c++ make%s\n" "$CYAN" "$NC"
            printf "\n"
            printf "Continue? (Y/n): "
            read -r -n 1 REPLY </dev/tty
            printf "\n"
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                sudo dnf install -y git python3 python3-pip python3-devel curl gcc gcc-c++ make
            fi
            ;;
        arch|manjaro|endeavouros)
            print_info "Using pacman package manager"
            printf "\n"
            printf "This will run: %ssudo pacman -Sy --noconfirm git python python-pip curl base-devel%s\n" "$CYAN" "$NC"
            printf "\n"
            printf "Continue? (Y/n): "
            read -r -n 1 REPLY </dev/tty
            printf "\n"
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                sudo pacman -Sy --noconfirm git python python-pip curl base-devel
            fi
            ;;
        opensuse*|suse*)
            print_info "Using zypper package manager"
            printf "\n"
            printf "This will run: %ssudo zypper install -y git python3 python3-pip curl gcc gcc-c++ make%s\n" "$CYAN" "$NC"
            printf "\n"
            printf "Continue? (Y/n): "
            read -r -n 1 REPLY </dev/tty
            printf "\n"
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                sudo zypper install -y git python3 python3-pip curl gcc gcc-c++ make
            fi
            ;;
        alpine)
            print_info "Using apk package manager"
            printf "\n"
            printf "This will run: %ssudo apk add git python3 py3-pip curl build-base%s\n" "$CYAN" "$NC"
            printf "\n"
            printf "Continue? (Y/n): "
            read -r -n 1 REPLY </dev/tty
            printf "\n"
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                sudo apk add git python3 py3-pip curl build-base
            fi
            ;;
        *)
            print_warning "Unknown Linux distribution: $distro"
            printf "\n"
            printf "Please manually install: git, python3, python3-pip, python3-venv, curl, and build tools\n"
            printf "\n"
            printf "Press Enter to continue..."
            read -r </dev/tty
            ;;
    esac
}

# Install Homebrew on macOS (optional, for Python installation)
install_homebrew() {
    if command_exists brew; then
        print_success "Homebrew is already installed"
        return 0
    fi
    
    printf "Homebrew is a package manager that can install Python and other tools.\n"
    printf "\n"
    printf "Install Homebrew? (Y/n): "
    read -r -n 1 REPLY </dev/tty
    printf "\n"
    
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        print_step "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add to PATH for Apple Silicon Macs
        if [ -f "/opt/homebrew/bin/brew" ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        
        print_success "Homebrew installed"
    fi
}

# Install Python via Homebrew on macOS
install_python_macos() {
    if ! command_exists brew; then
        install_homebrew
    fi
    
    if command_exists brew; then
        print_step "Installing Python via Homebrew..."
        brew install python3
        print_success "Python installed"
    else
        print_error "Cannot install Python without Homebrew"
        print_info "Please install Python manually from https://python.org"
        return 1
    fi
}

# Step 1: Welcome
step_welcome() {
    print_header "Welcome to Tau"
    
    printf "%sTau%s A chat agent that learns from you, writes its own software to upgrade itself, and solve problems for you. \n" "$BOLD" "$NC"
    printf "\n"
    printf "This installer will guide you through:\n"
    printf "\n"
    printf "  %sStep 1%s → Check & install system dependencies (git, python, dev tools)\n" "$CYAN" "$NC"
    printf "  %sStep 2%s → Install uv (Python package manager)\n" "$CYAN" "$NC"
    printf "  %sStep 3%s → Clone the Tau repository\n" "$CYAN" "$NC"
    printf "  %sStep 4%s → Set up Python environment & supervisor\n" "$CYAN" "$NC"
    printf "  %sStep 5%s → Install Cursor agent CLI\n" "$CYAN" "$NC"
    printf "  %sStep 6%s → Configure Telegram bot token\n" "$CYAN" "$NC"
    printf "  %sStep 7%s → Configure OpenAI API key (optional)\n" "$CYAN" "$NC"
    printf "  %sStep 8%s → Set up auto-start on login\n" "$CYAN" "$NC"
    printf "  %sStep 9%s → Launch Tau!\n" "$CYAN" "$NC"
    printf "\n"
    printf "%sSupported systems:%s macOS (Intel/Apple Silicon), Linux (Ubuntu, Debian, Fedora, Arch, etc.)\n" "$BOLD" "$NC"
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
    
    local OS=$(detect_os)
    local missing_git=false
    local missing_python=false
    local missing_curl=false
    
    # Check curl (needed for uv installation)
    print_step "Checking for curl..."
    if command_exists curl; then
        print_success "curl is installed"
    else
        print_error "curl is not installed"
        missing_curl=true
    fi
    
    # Check git
    print_step "Checking for git..."
    if command_exists git; then
        print_success "git is installed ($(git --version | cut -d' ' -f3))"
    else
        print_error "git is not installed"
        missing_git=true
    fi
    
    # Check Python
    print_step "Checking for Python 3..."
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION is installed"
    else
        print_error "Python 3 is not installed"
        missing_python=true
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
    
    # Handle missing dependencies
    if [ "$missing_git" = true ] || [ "$missing_python" = true ] || [ "$missing_curl" = true ]; then
        echo ""
        print_warning "Missing some required dependencies"
        printf "\n"
        
        if [ "$OS" = "macos" ]; then
            printf "On %smacOS%s, we can install developer tools automatically.\n" "$BOLD" "$NC"
            printf "\n"
            printf "Install missing dependencies? (Y/n): "
            read -r -n 1 REPLY </dev/tty
            printf "\n"
            
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                # Install Xcode Command Line Tools (provides git, curl, etc.)
                if [ "$missing_git" = true ] || [ "$missing_curl" = true ]; then
                    install_macos_dev_tools
                fi
                
                # Install Python if missing
                if [ "$missing_python" = true ]; then
                    install_python_macos
                fi
                
                # Re-check dependencies
                printf "\n"
                print_step "Re-checking dependencies..."
                
                if ! command_exists git; then
                    print_error "git is still not installed"
                else
                    print_success "git is now available"
                fi
                
                if ! command_exists python3; then
                    print_error "Python 3 is still not installed"
                else
                    print_success "Python 3 is now available"
                fi
                
                if ! command_exists curl; then
                    print_error "curl is still not installed"
                else
                    print_success "curl is now available"
                fi
            fi
            
        elif [ "$OS" = "linux" ]; then
            printf "On %sLinux%s, we can install dependencies using your package manager.\n" "$BOLD" "$NC"
            printf "\n"
            printf "Install missing dependencies? (Y/n): "
            read -r -n 1 REPLY </dev/tty
            printf "\n"
            
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                install_linux_dependencies
                
                # Re-check dependencies
                printf "\n"
                print_step "Re-checking dependencies..."
                
                if ! command_exists git; then
                    print_error "git is still not installed"
                else
                    print_success "git is now available"
                fi
                
                if ! command_exists python3; then
                    print_error "Python 3 is still not installed"
                else
                    print_success "Python 3 is now available"
                fi
                
                if ! command_exists curl; then
                    print_error "curl is still not installed"
                else
                    print_success "curl is now available"
                fi
            fi
        else
            print_error "Unknown operating system. Please install git, python3, and curl manually."
        fi
        
        # Final check - exit if still missing critical dependencies
        if ! command_exists git || ! command_exists python3 || ! command_exists curl; then
            echo ""
            print_error "Missing critical dependencies. Cannot continue."
            printf "\n"
            printf "Please manually install:\n"
            if ! command_exists git; then printf "  - git\n"; fi
            if ! command_exists python3; then printf "  - python3\n"; fi
            if ! command_exists curl; then printf "  - curl\n"; fi
            printf "\n"
            exit 1
        fi
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
        
        # Check if critical files are missing - force update if so
        if [ ! -f "$INSTALL_DIR/tauctl" ] || [ ! -f "$INSTALL_DIR/supervisord.conf" ]; then
            print_warning "Missing critical files - updating required"
            print_step "Updating repository..."
            cd "$INSTALL_DIR"
            git pull origin main || print_warning "Could not pull updates"
        else
            printf "Update existing installation? (Y/n): "
            read -r -n 1 REPLY </dev/tty
            printf "\n"
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                print_step "Updating repository..."
                cd "$INSTALL_DIR"
                git pull origin main || print_warning "Could not pull updates"
            else
                print_info "Using existing installation"
            fi
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
    
    print_step "Installing supervisor (process manager)..."
    uv pip install supervisor
    print_success "Supervisor installed"
    
    # Create logs directory
    mkdir -p "$INSTALL_DIR/logs"
    
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
    printf "This will install the Cursor agent CLI from %shttps://cursor.com/install%s\n" "$CYAN" "$NC"
    printf "\n"
    
    printf "Install Cursor agent CLI? (Y/n): "
    read -r -n 1 REPLY </dev/tty
    printf "\n"
    
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        print_step "Installing Cursor agent CLI..."
        printf "\n"
        
        curl https://cursor.com/install -fsSL | bash
        
        # Add common install locations to PATH for current session
        export PATH="$HOME/.cursor/bin:$HOME/.local/bin:$PATH"
        
        printf "\n"
        if command_exists agent; then
            print_success "Cursor agent CLI installed successfully!"
            
            printf "\n"
            print_step "Logging in to Cursor agent..."
            printf "\n"
            printf "This will open a browser window for authentication.\n"
            printf "\n"
            
            agent login
            
            printf "\n"
            print_success "Cursor agent login complete!"
        else
            print_warning "Cursor agent CLI installed but not found in PATH."
            print_info "You may need to restart your terminal or add it to your PATH."
            print_info "Then run: agent login"
        fi
    else
        print_info "Skipped. You can install it later with:"
        printf "  %scurl https://cursor.com/install -fsSL | bash%s\n" "$CYAN" "$NC"
        printf "  %sagent login%s\n" "$CYAN" "$NC"
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

# Step 9: Setup startup on login
step_setup_startup() {
    print_header "Step 8: Setting Up Auto-Start"
    
    local OS=$(detect_os)
    
    printf "Tau can start automatically when you log in.\n"
    printf "This uses %ssupervisord%s to manage the process.\n" "$BOLD" "$NC"
    printf "\n"
    printf "Benefits:\n"
    printf "  • Tau restarts automatically if it crashes\n"
    printf "  • Starts automatically on login\n"
    printf "  • Easy control with %stauctl%s command\n" "$CYAN" "$NC"
    printf "\n"
    
    printf "Enable auto-start on login? (Y/n): "
    read -r -n 1 REPLY </dev/tty
    printf "\n"
    
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_info "Skipped auto-start setup."
        print_info "You can run Tau manually with: tauctl start"
        wait_continue
        return
    fi
    
    if [ "$OS" = "macos" ]; then
        setup_launchd_startup
    elif [ "$OS" = "linux" ]; then
        setup_systemd_startup
    else
        print_warning "Auto-start not supported on this OS."
        print_info "Run Tau manually with: tauctl start"
    fi
    
    wait_continue
}

# Setup launchd on macOS
setup_launchd_startup() {
    local PLIST_DIR="$HOME/Library/LaunchAgents"
    local PLIST_FILE="$PLIST_DIR/com.tau.supervisor.plist"
    
    mkdir -p "$PLIST_DIR"
    
    print_step "Creating LaunchAgent for supervisord..."
    
    cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.tau.supervisor</string>
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
        <key>PATH</key>
        <string>$INSTALL_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF
    
    print_success "LaunchAgent created at $PLIST_FILE"
    
    # Load the agent
    print_step "Loading LaunchAgent..."
    launchctl unload "$PLIST_FILE" 2>/dev/null || true
    launchctl load "$PLIST_FILE"
    
    print_success "Tau will now start automatically on login!"
    printf "\n"
    print_info "To disable auto-start later:"
    printf "  %slaunchctl unload %s%s\n" "$CYAN" "$PLIST_FILE" "$NC"
}

# Setup systemd on Linux
setup_systemd_startup() {
    local SERVICE_DIR="$HOME/.config/systemd/user"
    local SERVICE_FILE="$SERVICE_DIR/tau-supervisor.service"
    
    mkdir -p "$SERVICE_DIR"
    
    print_step "Creating systemd user service..."
    
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Tau Supervisord
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
    
    print_success "Service created at $SERVICE_FILE"
    
    # Reload and enable
    print_step "Enabling systemd service..."
    systemctl --user daemon-reload
    systemctl --user enable tau-supervisor
    systemctl --user start tau-supervisor
    
    # Enable lingering so user services start at boot
    print_step "Enabling user service persistence..."
    loginctl enable-linger "$USER" 2>/dev/null || print_warning "Could not enable linger (may need sudo)"
    
    print_success "Tau will now start automatically on login!"
    printf "\n"
    print_info "To disable auto-start later:"
    printf "  %ssystemctl --user disable tau-supervisor%s\n" "$CYAN" "$NC"
}

# Step 10: Launch Tau
step_launch_tau() {
    print_header "Step 9: Launching Tau!"
    
    printf "%sInstallation complete!%s\n" "$GREEN" "$NC"
    printf "\n"
    printf "Tau is managed by %ssupervisord%s for reliable operation.\n" "$BOLD" "$NC"
    printf "\n"
    printf "%sControl commands:%s\n" "$BOLD" "$NC"
    printf "\n"
    printf "  %stauctl start%s    - Start tau\n" "$CYAN" "$NC"
    printf "  %stauctl stop%s     - Stop tau\n" "$CYAN" "$NC"
    printf "  %stauctl restart%s  - Restart tau\n" "$CYAN" "$NC"
    printf "  %stauctl status%s   - Check status\n" "$CYAN" "$NC"
    printf "  %stauctl logs%s     - View logs\n" "$CYAN" "$NC"
    printf "  %stauctl logs -f%s  - Follow logs\n" "$CYAN" "$NC"
    printf "\n"
    printf "%sTelegram commands:%s\n" "$BOLD" "$NC"
    printf "\n"
    printf "  %s/start%s     - Initialize the bot\n" "$CYAN" "$NC"
    printf "  %s/task%s      - Add a new task\n" "$CYAN" "$NC"
    printf "  %s/status%s    - Check current status\n" "$CYAN" "$NC"
    printf "  %s/adapt%s     - Self-modify the bot\n" "$CYAN" "$NC"
    printf "  %s/restart%s   - Restart the bot\n" "$CYAN" "$NC"
    printf "\n"
    
    # Add tauctl to PATH info
    printf "%sTo use tauctl from anywhere:%s\n" "$BOLD" "$NC"
    printf "\n"
    printf "  Add to your shell config (~/.zshrc or ~/.bashrc):\n"
    printf "  %sexport PATH=\"%s:\$PATH\"%s\n" "$CYAN" "$INSTALL_DIR" "$NC"
    printf "\n"
    
    # Check if tau is already running via supervisor
    if [ -f "$INSTALL_DIR/.supervisord.pid" ]; then
        local pid=$(cat "$INSTALL_DIR/.supervisord.pid" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            print_success "Tau is already running via supervisord!"
            printf "\n"
            printf "Open Telegram and message your bot to get started.\n"
            printf "\n"
            return
        fi
    fi
    
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
        
        # Start via tauctl
        "$INSTALL_DIR/tauctl" start
        
        printf "\n"
        print_success "Tau is running!"
        printf "\n"
        printf "Open Telegram and message your bot to get started.\n"
        printf "Use %stauctl logs -f%s to follow the logs.\n" "$CYAN" "$NC"
    else
        printf "\n"
        print_info "To start Tau later, run:"
        printf "\n"
        printf "  %scd %s && ./tauctl start%s\n" "$CYAN" "$INSTALL_DIR" "$NC"
        printf "\n"
    fi
}

# Main installation flow
main() {
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
    step_setup_startup
    step_launch_tau
}

# Run main function
main "$@"
