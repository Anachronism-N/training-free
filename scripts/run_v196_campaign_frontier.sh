#!/usr/bin/env bash
# Inspect and package the v189-v195 evidence frontier without running a GPU job.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    inspect|show|next|package) ;;
    *)
        echo "usage: bash scripts/run_v196_campaign_frontier.sh {inspect|show|next|package}"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_ROOT="${V196_OUT_ROOT:-$ROOT/runs/v196_campaign_frontier}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_inspect() {
    "$PYTHON_BIN" "$ROOT/scripts/inspect_v196_campaign_frontier.py" inspect \
        --repo-root "$ROOT" --output-root "$OUT_ROOT"
}

case "$ACTION" in
    inspect)
        run_inspect
        ;;
    show)
        run_inspect
        cat "$OUT_ROOT/campaign_state.md"
        ;;
    next)
        run_inspect
        cat "$OUT_ROOT/next_commands.txt"
        ;;
    package)
        "$PYTHON_BIN" "$ROOT/scripts/inspect_v196_campaign_frontier.py" package \
            --repo-root "$ROOT" --output-root "$OUT_ROOT"
        ;;
esac
