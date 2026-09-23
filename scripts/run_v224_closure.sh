#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
V219_OUT_ROOT=${V219_OUT_ROOT:-/apdcephfs_gy2/share_302533218/cedricnie/v219_runs/v219_38a06347_mechanism64}
V220_OUT_ROOT=${V220_OUT_ROOT:-/apdcephfs_gy2/share_302533218/cedricnie/v220_runs/v220_1a1c0885_long60_64}
V223_OUT_ROOT=${V223_OUT_ROOT:-/apdcephfs_gy2/share_302533218/cedricnie/v223_runs/v223_core32}
V224_OUT_ROOT=${V224_OUT_ROOT:-/apdcephfs_gy2/share_302533218/cedricnie/v224_runs/closure}
NUM_NODES=${NUM_NODES:-8}

case "${1:-}" in
  export)
    exec "$PYTHON" "$ROOT/scripts/export_v224_submission_evidence.py" \
      --v219-root "$V219_OUT_ROOT" --v220-root "$V220_OUT_ROOT" \
      --v223-root "$V223_OUT_ROOT" --output-root "$V224_OUT_ROOT/paper_evidence"
    ;;
  audit)
    : "${NODE_RANK:?Set NODE_RANK to a distinct rank 0..7, or 0 with NUM_NODES=1}"
    exec "$PYTHON" "$ROOT/scripts/audit_v224_reused_clips.py" run \
      --v219-root "$V219_OUT_ROOT" --v223-root "$V223_OUT_ROOT" \
      --output-root "$V224_OUT_ROOT" --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" \
      --workers "${CPU_WORKERS:-4}" --ffmpeg "${FFMPEG:-ffmpeg}"
    ;;
  collect)
    exec "$PYTHON" "$ROOT/scripts/audit_v224_reused_clips.py" collect \
      --v223-root "$V223_OUT_ROOT" --output-root "$V224_OUT_ROOT" --num-nodes "$NUM_NODES"
    ;;
  *)
    printf 'Usage: bash scripts/run_v224_closure.sh {export|audit|collect}\n' >&2
    exit 2
    ;;
esac
