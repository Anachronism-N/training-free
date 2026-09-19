#!/usr/bin/env bash
set -euo pipefail
export LPHC_CAMPAIGN=v216
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_v212_experiment.sh" "$@"
