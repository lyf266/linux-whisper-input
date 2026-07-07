#!/bin/bash
# Automated installation script for Arch Linux Voice Input Service
# Optimized for quick, one-key setup by users or developer agents.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== 1. Installing System Dependencies ==="
echo "This requires sudo privileges to install system packages via pacman."
sudo pacman -S --needed --noconfirm portaudio notify-send wl-clipboard xdotool ydotool

echo "=== 2. Configuring User Permissions for ydotool ==="
# Add current user to input group for virtual keyboard injection
if ! groups | grep -q "\binput\b"; then
    echo "Adding user $USER to the 'input' group..."
    sudo usermod -aG input "$USER"
    echo "⚠️ Note: You will need to log out and log back in for input group permissions to take effect!"
fi

# Enable ydotool systemd user service
echo "Enabling and starting ydotool user service..."
systemctl --user enable --now ydotool.service || true

# Set up udev rules and modules for virtual inputs
echo "Setting up uinput modules and udev rules..."
echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' | sudo tee /etc/udev/rules.d/80-uinput.rules > /dev/null
echo 'uinput' | sudo tee /etc/modules-load.d/uinput.conf > /dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger || true

echo "=== 3. Setting up Python Virtual Environment ==="
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with --system-site-packages..."
    python3 -m venv --system-site-packages venv
fi
echo "Installing requirements..."
./venv/bin/pip install -r requirements.txt

echo "=== 4. Downloading large-v3-turbo Model from ModelScope ==="
# Bypasses HuggingFace GFW CDN blocks in China using direct ModelScope download
./venv/bin/python download_model_modelscope.py

echo "=== 5. Registering and Enabling Systemd User Daemon ==="
mkdir -p ~/.config/systemd/user/
ln -sf "$DIR/voice-input.service" ~/.config/systemd/user/voice-input.service
systemctl --user daemon-reload
systemctl --user enable --now voice-input.service

echo "=== 6. Binding KDE Global Shortcut (Meta+H) ==="
./venv/bin/python set_shortcut.py || echo "⚠️ Warning: Failed to register shortcut via DBus. You can set it manually in KDE System Settings."

echo "=================================================="
echo "🎉 Installation Completed Successfully!"
echo "If this is your first time setting up ydotool, please LOG OUT and LOG IN again."
echo "You can check the daemon status with:"
echo "  systemctl --user status voice-input.service"
echo "=================================================="
