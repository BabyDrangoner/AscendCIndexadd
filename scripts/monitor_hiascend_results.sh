#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE_BIN="${NODE_BIN:-}"

if [[ -z "$NODE_BIN" ]]; then
  if command -v node >/dev/null 2>&1; then
    NODE_BIN="$(command -v node)"
  elif [[ -x "/Users/xiaolong.xu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node" ]]; then
    NODE_BIN="/Users/xiaolong.xu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
  else
    echo "node not found; set NODE_BIN=/path/to/node" >&2
    exit 1
  fi
fi

exec "$NODE_BIN" "$ROOT_DIR/scripts/monitor_hiascend_results.mjs" "$@"
