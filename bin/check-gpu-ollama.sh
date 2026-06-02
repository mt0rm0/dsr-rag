#!/usr/bin/env bash
# Opens a tmux session showing GPU usage alongside the Ollama service log.
# Useful for monitoring GPU utilisation during the workshop.
# Requires: tmux, ollama (running as a systemd service)

set -euo pipefail

tmux new -s ollama -d
tmux splitw -v -t ollama

if type nvtop > /dev/null 2>&1; then
    tmux send-keys -t ollama:0.0 'nvtop' Enter
else
    tmux send-keys -t ollama:0.0 'watch -n 1 nvidia-smi' Enter
fi

tmux send-keys -t ollama:0.1 'journalctl -xefu ollama.service' Enter
tmux attach -t ollama
