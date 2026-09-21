#!/usr/bin/env bash
set -euo pipefail
export LPHC_CAMPAIGN=v219
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_v212_experiment.sh" "$@"
