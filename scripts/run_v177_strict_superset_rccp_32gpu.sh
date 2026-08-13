#!/usr/bin/env bash
# v177 fixes v176's teacher eligibility boundary and makes subset audit fatal.
set -euo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
export PROFILE_CONTRACT=v177
export PROFILE_VERSION=3
export EXPERIMENT_NAME=v177_strict_superset_rccp
# Keep v176's frozen split. Re-shuffling after seeing an invalid run would
# contaminate the previously untouched generation holdout.
export DISCOVERY_SEED=1762026
export RUN_LABEL=v177
export V176_SOURCE_PROMPTS="${V177_SOURCE_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
export V176_OUT_ROOT="${V177_OUT_ROOT:-$ROOT/runs/v177_strict_superset_rccp}"

exec bash "$ROOT/scripts/run_v176_superset_rccp_32gpu.sh" "$@"
