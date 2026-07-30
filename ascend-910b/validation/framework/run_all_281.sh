#!/usr/bin/env bash
# One command to create and submit the complete archived-bundle queue.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python3 generate_inventory_manifest.py
python3 build_test_mapping.py
python3 materialize_full_queue.py
python3 run_ascend_validation.py --manifest full-execution-queue.json --execute
