#!/usr/bin/env bash
# run_loo_lr_grid.sh
# ───────────────────────────────────────────────────────────
# Grid search: 8 learning-rates  ×  9 leave-one-out subjects
# Adjust the arrays or paths as needed before executing.
# Make executable:  chmod +x run_loo_lr_grid.sh
# Run:             ./run_loo_lr_grid.sh
# ───────────────────────────────────────────────────────────

# ── 1.  Hyper-parameters ──────────────────────────────────
lrs=(5e-3 1e-4 5e-4 1e-5 5e-5)

subjects=(evan helen jack jason jin kanav maggie michelle vidya)

# ── 2.  Loop over subjects × learning-rates ───────────────
for lr in "${lrs[@]}"; do
    exp_tag="loo_no_pretrain_${lr}"
    rm -rf ./pose_models/checkpoints/${exp_tag}

  for subject in "${subjects[@]}"; do

    # Unique experiment tags keep runs tidy
    echo "▶ Subject: $subject   LR: $lr"

    # ── Train ──────────────────────────────────────────────
    python scripts/train.py \
      --experiment "${exp_tag}" \
      --model "imu_uwb_pose_model" \
      --dataset "footposer_dataset" \
      --lr "${lr}" \
      --loo "${subject}"

    # ── Test ───────────────────────────────────────────────
    python scripts/test.py \
      --experiment "${exp_tag}" \
      --dataset "footposer_dataset" \
      --loo "${subject}"

    echo "✓ Done  $subject  @ lr=$lr"
    echo "------------------------------------------------------"
  done
done