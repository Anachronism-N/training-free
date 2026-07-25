#!/usr/bin/env bash
# End-to-end v96: QK profiling, 16-cell generation, then metrics.
set -euo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
export REPO_ROOT="$ROOT"

bash "$ROOT/scripts/run_v96_qk_head_profile_16gpu.sh"
bash "$ROOT/scripts/run_v96_binary_cache_16gpu.sh"
bash "$ROOT/scripts/postprocess_v96_binary_cache.sh"
