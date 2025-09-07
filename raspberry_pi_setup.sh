#!/bin/bash

# T8 Delays Monitor - Raspberry Pi Setup Script
# This script automates the installation and setup of the T8 monitor on Raspberry Pi

set -e  # Exit on any error

echo "🚀 Starting T8 Delays Monitor setup for Raspberry Pi..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Run as regular user (pi)."
   exit 1
fi

# Update system packages
print_info "Updating system packages..."
sudo apt update && sudo apt upgrade -y
print_status "System packages updated"

# Install system dependencies
print_info "Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libffi-dev \
    libssl-dev
print_status "System dependencies installed"

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_info "Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 0 ]]; then
    print_error "Python 3.8+ required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create project directory
PROJECT_DIR="$HOME/t8-monitor"
print_info "Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create virtual environment
print_info "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
print_status "Virtual environment created and activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip
print_status "Pip upgraded"

# Install Python dependencies
print_info "Installing Python dependencies..."
cat > requirements.txt << EOF
tweepy>=4.14.0
python-telegram-bot>=20.0
python-dateutil>=2.8.0
ollama
aiohttp>=3.8.0
requests>=2.25.0
holidays>=0.34
icalendar>=5.0.0
beautifulsoup4>=4.9.0
EOF

pip install -r requirements.txt
print_status "Python dependencies installed"

# Install Docker (for Ollama)
print_info "Installing Docker for Ollama..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    print_status "Docker installed"
    print_warning "You may need to log out and back in for Docker permissions to take effect"
else
    print_status "Docker already installed"
fi

# Install Ollama
print_info "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    print_status "Ollama installed"
else
    print_status "Ollama already installed"
fi

# Start Ollama service
print_info "Starting Ollama service..."
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for Ollama to be ready
print_info "Waiting for Ollama to be ready..."
sleep 10

# Pull the required model
print_info "Pulling Ollama model (llama3.2:3b)..."
ollama pull llama3.2:3b
print_status "Ollama model downloaded"

# Create systemd service
print_info "Creating systemd service..."
sudo tee /etc/systemd/system/t8-monitor.service > /dev/null << EOF
[Unit]
Description=T8 Delays Monitor
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python monitor_t8_delays_polling.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
print_status "Systemd service created"

# Create log rotation
print_info "Setting up log rotation..."
sudo tee /etc/logrotate.d/t8-monitor > /dev/null << EOF
$PROJECT_DIR/t8_monitor.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 $USER $USER
}
EOF
print_status "Log rotation configured"

# Create environment file template
print_info "Creating environment configuration template..."
cat > .env.example << 'EOF'
# Choose your API backend (set to true for TwitterAPI.io, false for X API)
USE_TWITTERAPI_IO=true

# TwitterAPI.io configuration (if USE_TWITTERAPI_IO=true)
TWITTERAPI_IO_KEY=your_twitterapi_io_key_here

# X API configuration (if USE_TWITTERAPI_IO=false)  
X_BEARER_TOKEN=your_x_bearer_token_here

# Telegram configuration (required)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Ollama configuration (optional)
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434

# Polling interval in minutes (default: 60)
# For TwitterAPI.io: 2-5 minutes for high frequency, 30-60 for quota-friendly
# For X API: 60+ minutes recommended to stay within free tier limits
POLLING_INTERVAL_MINUTES=60

# Debug logging (optional)
DEBUG=false
EOF

print_status "Environment template created"

print_info "Setup completed! Next steps:"
echo ""
echo "1. Copy your Python files to $PROJECT_DIR:"
echo "   - monitor_t8_delays_polling.py"
echo "   - sydney_school_day_checker.py"
echo ""
echo "2. Configure your environment:"
echo "   cp .env.example .env"
echo "   nano .env  # Edit with your API keys"
echo ""
echo "3. Test the setup:"
echo "   cd $PROJECT_DIR"
echo "   source venv/bin/activate"
echo "   python monitor_t8_delays_polling.py"
echo ""
echo "4. Enable auto-start:"
echo "   sudo systemctl enable t8-monitor"
echo "   sudo systemctl start t8-monitor"
echo ""
echo "5. Check status:"
echo "   sudo systemctl status t8-monitor"
echo "   journalctl -u t8-monitor -f"
echo ""
print_status "Setup script completed successfully!"
