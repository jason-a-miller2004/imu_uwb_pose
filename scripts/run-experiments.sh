#!/bin/bash

# Define an array of learning rates
learning_rates=(3e-6 3e-7)

# Loop through each learning rate
for lr in "${learning_rates[@]}"
do
  echo "Running experiment with learning rate $lr"
  python scripts/train.py --experiment "pretrain_run-lr=$lr" --lr=$lr
done
