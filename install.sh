#!/usr/bin/env bash
#
# Tau Installation Script
# 
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/Tau/main/install.sh | bash
#
# Or clone and run locally:
#   ./install.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Default installation directory
INSTALL_DIR="${TAU_INSTALL_DIR:-$HOME/Tau}"
REPO_URL="${TAU_REPO_URL:-https://github.com/YOUR_USERNAME/Tau.git}"

# Print colored output
print_header() {
    echo -e "\n${BOLD}${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${NC}\n"
}

print_step() {
    echo -e "${BLUE}→${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
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

# Check for required dependencies
check_dependencies() {
    print_header "Checking Dependencies"
    
    local missing_deps=()
    
    # Check git
    if command_exists git; then
        print_success "git is installed"
    else
        missing_deps+=("git")
        print_error "git is not installed"
    fi
    
    # Check Python
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION is installed"
    else
        missing_deps+=("python3")
        print_error "Python 3 is not installed"
    fi
    
    # Check uv (Python package manager)
    if command_exists uv; then
        print_success "uv is installed"
    else
        print_warning "uv is not installed (will install)"
    fi
    
    # Check Cursor agent CLI
    if command_exists agent; then
        print_success "Cursor agent CLI is installed"
    else
        print_warning "Cursor agent CLI is not installed (will guide installation)"
    fi
    
    # Exit if critical dependencies are missing
    if [[ " ${missing_deps[*]} " =~ " git " ]] || [[ " ${missing_deps[*]} " =~ " python3 " ]]; then
        echo ""
        print_error "Missing critical dependencies. Please install:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        exit 1
    fi
}

# Install uv if not present
install_uv() {
    if ! command_exists uv; then
        print_step "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # Add to PATH for current session
        export PATH="$HOME/.local/bin:$PATH"
        
        if command_exists uv; then
            print_success "uv installed successfully"
        else
            print_error "Failed to install uv"
            exit 1
        fi
    fi
}

# Clone or update the repository
clone_repo() {
    print_header "Setting Up Tau Repository"
    
    if [ -d "$INSTALL_DIR" ]; then
        print_info "Tau directory already exists at $INSTALL_DIR"
        read -p "Update existing installation? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_step "Updating repository..."
            cd "$INSTALL_DIR"
            git pull origin main || print_warning "Could not pull updates (may be local changes)"
        fi
    else
        print_step "Cloning Tau repository to $INSTALL_DIR..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        print_success "Repository cloned successfully"
    fi
    
    cd "$INSTALL_DIR"
}

# Set up Python virtual environment and install dependencies
setup_python() {
    print_header "Setting Up Python Environment"
    
    cd "$INSTALL_DIR"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        print_step "Creating Python virtual environment..."
        uv venv .venv
        print_success "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install dependencies
    print_step "Installing Python dependencies..."
    uv pip install -e .
    print_success "Dependencies installed"
}

# Guide user through Cursor agent installation
setup_cursor_agent() {
    print_header "Setting Up Cursor Agent CLI"
    
    if command_exists agent; then
        print_success "Cursor agent CLI is already installed"
        return
    fi
    
    echo -e "${YELLOW}The Cursor agent CLI is required for Tau to function.${NC}"
    echo ""
    echo "To install it:"
    echo ""
    echo "  1. Open Cursor IDE"
    echo "  2. Press ${BOLD}Cmd+Shift+P${NC} (macOS) or ${BOLD}Ctrl+Shift+P${NC} (Linux)"
    echo "  3. Type: ${BOLD}Install 'agent' command${NC}"
    echo "  4. Select the option to install the CLI"
    echo ""
    echo "Alternatively, if you have Cursor installed, try:"
    echo "  ${CYAN}cursor --install-extension cursor.agent${NC}"
    echo ""
    
    read -p "Press Enter once you've installed the Cursor agent CLI..."
    
    # Verify installation
    if command_exists agent; then
        print_success "Cursor agent CLI is now installed"
    else
        print_warning "Cursor agent CLI not detected. Tau may not function correctly."
        print_info "You can continue and install it later."
    fi
}

# Set up Telegram bot token
setup_telegram_token() {
    print_header "Setting Up Telegram Bot Token"
    
    echo "Tau uses a Telegram bot to communicate with you."
    echo ""
    echo "To create a Telegram bot:"
    echo ""
    echo "  1. Open Telegram and search for ${BOLD}@BotFather${NC}"
    echo "  2. Send ${BOLD}/newbot${NC} to create a new bot"
    echo "  3. Follow the prompts to name your bot"
    echo "  4. Copy the ${BOLD}HTTP API token${NC} BotFather gives you"
    echo ""
    
    # Check if token is already set
    if [ -n "$TAU_BOT_TOKEN" ]; then
        print_info "TAU_BOT_TOKEN is already set in environment"
        read -p "Use existing token? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            return
        fi
    fi
    
    read -p "Enter your Telegram Bot Token (or press Enter to skip): " BOT_TOKEN
    
    if [ -n "$BOT_TOKEN" ]; then
        # Create .env file
        ENV_FILE="$INSTALL_DIR/.env"
        
        # Check if .env exists and has the token
        if [ -f "$ENV_FILE" ] && grep -q "TAU_BOT_TOKEN" "$ENV_FILE"; then
            # Update existing token
            if [[ "$(uname -s)" == "Darwin" ]]; then
                sed -i '' "s/^TAU_BOT_TOKEN=.*/TAU_BOT_TOKEN=$BOT_TOKEN/" "$ENV_FILE"
            else
                sed -i "s/^TAU_BOT_TOKEN=.*/TAU_BOT_TOKEN=$BOT_TOKEN/" "$ENV_FILE"
            fi
        else
            # Append token
            echo "TAU_BOT_TOKEN=$BOT_TOKEN" >> "$ENV_FILE"
        fi
        
        print_success "Token saved to $ENV_FILE"
        
        # Also suggest adding to shell profile
        echo ""
        print_info "To use Tau from anywhere, add this to your shell profile:"
        echo ""
        echo "  ${CYAN}export TAU_BOT_TOKEN=\"$BOT_TOKEN\"${NC}"
        echo ""
    else
        print_warning "Skipping token setup. Set TAU_BOT_TOKEN before running Tau."
    fi
}

# Set up OpenAI API key (optional)
setup_openai_key() {
    print_header "Setting Up OpenAI API Key (Optional)"
    
    echo "OpenAI API key enables voice message transcription."
    echo "This is ${BOLD}optional${NC} - Tau works without it."
    echo ""
    
    if [ -n "$OPENAI_API_KEY" ]; then
        print_info "OPENAI_API_KEY is already set in environment"
        return
    fi
    
    read -p "Enter your OpenAI API Key (or press Enter to skip): " API_KEY
    
    if [ -n "$API_KEY" ]; then
        ENV_FILE="$INSTALL_DIR/.env"
        
        if [ -f "$ENV_FILE" ] && grep -q "OPENAI_API_KEY" "$ENV_FILE"; then
            if [[ "$(uname -s)" == "Darwin" ]]; then
                sed -i '' "s/^OPENAI_API_KEY=.*/OPENAI_API_KEY=$API_KEY/" "$ENV_FILE"
            else
                sed -i "s/^OPENAI_API_KEY=.*/OPENAI_API_KEY=$API_KEY/" "$ENV_FILE"
            fi
        else
            echo "OPENAI_API_KEY=$API_KEY" >> "$ENV_FILE"
        fi
        
        print_success "API key saved to $ENV_FILE"
    else
        print_info "Skipping OpenAI setup. Voice features will be disabled."
    fi
}

# Create startup script
create_startup_script() {
    print_header "Creating Startup Script"
    
    START_SCRIPT="$INSTALL_DIR/start.sh"
    
    cat > "$START_SCRIPT" << 'EOF'
#!/usr/bin/env bash
# Start Tau agent

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Activate virtual environment
source .venv/bin/activate

# Run Tau
python -m tau
EOF
    
    chmod +x "$START_SCRIPT"
    print_success "Created start.sh"
    
    # Create systemd service for Linux (optional)
    if [[ "$(detect_os)" == "linux" ]]; then
        print_info "On Linux, you can run Tau as a systemd service."
        print_info "Run: ./install.sh --systemd to set this up."
    fi
}

# Create systemd service (Linux only)
create_systemd_service() {
    if [[ "$(detect_os)" != "linux" ]]; then
        print_error "Systemd services are only available on Linux"
        return
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
    echo "To enable and start Tau as a service:"
    echo "  ${CYAN}systemctl --user daemon-reload${NC}"
    echo "  ${CYAN}systemctl --user enable tau${NC}"
    echo "  ${CYAN}systemctl --user start tau${NC}"
    echo ""
    echo "To check status:"
    echo "  ${CYAN}systemctl --user status tau${NC}"
}

# Print final instructions
print_final_instructions() {
    print_header "Installation Complete!"
    
    echo -e "${GREEN}Tau has been installed successfully!${NC}"
    echo ""
    echo "Installation directory: ${BOLD}$INSTALL_DIR${NC}"
    echo ""
    echo "${BOLD}To start Tau:${NC}"
    echo ""
    echo "  ${CYAN}cd $INSTALL_DIR${NC}"
    echo "  ${CYAN}./start.sh${NC}"
    echo ""
    echo "${BOLD}Or run directly:${NC}"
    echo ""
    echo "  ${CYAN}cd $INSTALL_DIR && source .venv/bin/activate && python -m tau${NC}"
    echo ""
    echo "${BOLD}First time setup:${NC}"
    echo ""
    echo "  1. Start Tau with one of the commands above"
    echo "  2. Open Telegram and search for your bot"
    echo "  3. Send ${BOLD}/start${NC} to initialize"
    echo ""
    echo "${BOLD}Available Telegram commands:${NC}"
    echo ""
    echo "  /start     - Initialize the bot"
    echo "  /task      - Add a new task"
    echo "  /status    - Check current status"
    echo "  /adapt     - Self-modify the bot"
    echo "  /restart   - Restart the bot"
    echo ""
    print_info "For more information, see: $INSTALL_DIR/README.md"
}

# Main installation flow
main() {
    print_header "Tau Installation"
    
    echo "This script will install Tau, a self-adapting autonomous agent"
    echo "with Telegram integration."
    echo ""
    
    # Handle --systemd flag
    if [[ "$1" == "--systemd" ]]; then
        create_systemd_service
        exit 0
    fi
    
    # Handle custom install directory
    if [[ -n "$1" ]] && [[ "$1" != --* ]]; then
        INSTALL_DIR="$1"
    fi
    
    echo "Installation directory: ${BOLD}$INSTALL_DIR${NC}"
    echo ""
    read -p "Continue with installation? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    
    # Run installation steps
    check_dependencies
    install_uv
    clone_repo
    setup_python
    setup_cursor_agent
    setup_telegram_token
    setup_openai_key
    create_startup_script
    print_final_instructions
}

# Run main function
main "$@"
