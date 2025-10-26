#!/usr/bin/env bash
set -euo pipefail

# --- Check for GPU mode ---
if [ -z "${TIDL_TOOLS_TYPE:-}" ]; then
    echo "TIDL_TOOLS_TYPE unset, defaulting to CPU tools"
    tidl_gpu_tools=0
else
    if [ "$TIDL_TOOLS_TYPE" == "GPU" ]; then
        tidl_gpu_tools=1
    else
        tidl_gpu_tools=0
    fi
fi

echo "🔧 Setting up Docker environment on NixOS..."

# --- Check if Docker is available ---
if ! command -v docker &>/dev/null; then
    echo "Docker not found! Please enable it in your NixOS configuration.nix:"
    echo
    echo '  virtualisation.docker.enable = true;'
    echo '  users.users.jade.extraGroups = [ "docker" ];'
    echo
    echo "Then run: sudo nixos-rebuild switch"
    exit 1
fi

# --- Ensure your user is in docker group ---
if ! groups | grep -q docker; then
    echo "You are not in the docker group yet!"
    echo "Run the following, then log out and back in:"
    echo "  sudo usermod -aG docker $USER"
    exit 1
fi

# --- GPU Support (optional) ---
if [ "$tidl_gpu_tools" -eq 1 ]; then
    echo "GPU tools requested (TIDL_TOOLS_TYPE=GPU)"
    echo "Make sure your NixOS config includes NVIDIA container support."
    echo "Example:"
    echo
    echo '  virtualisation.docker.enableNvidia = true;'
    echo '  hardware.nvidia-container-toolkit.enable = true;'
    echo
    echo "Then run: sudo nixos-rebuild switch"
else
    echo "CPU tools selected. No extra setup needed."
fi

echo "Docker environment ready!"

