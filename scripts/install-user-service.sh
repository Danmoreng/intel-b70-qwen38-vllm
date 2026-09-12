#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
unit_file="$unit_dir/b70-qwen38-vllm.service"
mkdir -p "$unit_dir"
sed "s|@REPO_DIR@|$repo_dir|g" \
  "$repo_dir/systemd/b70-qwen38-vllm.service.in" > "$unit_file"
systemctl --user daemon-reload
systemctl --user enable --now b70-qwen38-vllm.service
systemctl --user status --no-pager b70-qwen38-vllm.service

