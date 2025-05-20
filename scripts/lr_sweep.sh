#!/usr/bin/env bash
# run_loo_lr_grid.sh
# ───────────────────────────────────────────────────────────
# Grid search: 8 learning-rates  ×  9 leave-one-out subjects
# Adjust the arrays or paths as needed before executing.
# Make executable:  chmod +x run_loo_lr_grid.sh
# Run:             ./run_loo_lr_grid.sh
# ───────────────────────────────────────────────────────────

# ── 1.  Hyper-parameters ──────────────────────────────────
lrs=(1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9)

subjects=(evan helen jack jason jin kanav maggie michelle vidya)

# ── 2.  Loop over subjects × learning-rates ───────────────
for lr in "${lrs[@]}"; do
  for subject in "${subjects[@]}"; do

    rm -rf ./pose_models/checkpoints/pretrain_run-lr=3e-7-finetune

    # Unique experiment tags keep runs tidy
    exp_tag="pretrain_run-lr=3e-7"
    echo "▶ Subject: $subject   LR: $lr"

    # ── Train ──────────────────────────────────────────────
    python scripts/train.py \
      --experiment "${exp_tag}" \
      --lr "${lr}" \
      --finetune "${subject}"

    # ── Test ───────────────────────────────────────────────
    python scripts/test.py \
      --experiment "${exp_tag}-finetune" \
      --lr "${lr}" \
      --finetune "${subject}"

    echo "✓ Done  $subject  @ lr=$lr"
    echo "------------------------------------------------------"
  done
done