#!/usr/bin/env bash
# File: run_all_tests.sh

CHECKPOINT_ROOT="./pose_models/checkpoints"

for dir in "${CHECKPOINT_ROOT}"/*/; do
  # Skip items that aren’t directories
  [[ -d "$dir" ]] || continue

  experiment_name="$(basename "$dir")"
  echo "Running: python scripts/test.py --experiment ${experiment_name} --finetune"
  python scripts/test.py --experiment "${experiment_name}" --finetune
done
