#!/bin/bash

subjects=("evan" "helen" "jack" "jason" "jin" "kanav" "maggie" "michelle" "vidya")

for subject in "${subjects[@]}"
do
    echo "Running LOO for subject: $subject"

    # Train command
    python scripts/train.py --experiment train_run_accel --finetune "$subject" --lr 3e-3

    # Test command
    python scripts/test.py --experiment train_run_accel-finetune --finetune "$subject" --lr 3e-3
done
