#!/bin/bash
set +e
export HOME=/home/mdeopuja
export REPO_ROOT=/shared_inference/mdeopuja/Hyperloom
export PATH="/home/mdeopuja/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
set -a; . "$REPO_ROOT/.env" 2>/dev/null; set +a
. "$REPO_ROOT/runtime/kernel-agent.env.sh" 2>/dev/null
echo "BASE_URL=$ANTHROPIC_BASE_URL  MODEL=${CLAUDE_MODEL}  headers_set=$([ -n \"$ANTHROPIC_CUSTOM_HEADERS\" ] && echo yes || echo no)"
echo "=== claude CLI live gateway probe (Claude-Opus-5) ==="
timeout 90 claude --model "${CLAUDE_MODEL:-Claude-Opus-5}" -p "Reply with exactly one word: pong" 2>&1 | head -20
echo "=== claude exit=$? ==="
