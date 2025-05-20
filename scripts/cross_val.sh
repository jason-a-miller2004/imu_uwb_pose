#!/bin/bash

subjects=("evan" "helen" "jack" "jason" "jin" "kanav" "maggie" "michelle" "vidya")

for subject in "${subjects[@]}"
do
    echo "Running LOO for subject: $subject"

    # Train command
    python scripts/train.py --experiment pretrain_run-lr=3e-7 --finetune "$subject" --lr 3e-3

    # Test command
    python scripts/test.py --experiment pretrain_run-lr=3e-7-finetune --finetune "$subject" --lr 3e-3
done
