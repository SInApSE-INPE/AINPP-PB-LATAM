#!/bin/bash

# Activate environment (adapt as needed or assume user has active env)
# source activate /scratch/ideeps/adriano.almeida/sinapse

# Run training with specific 1-year train and 1-month val periods
# Example: Train 2020, Val Jan 2021
python scripts/train.py \
    dataset.train_period.start="2020-01-01" \
    dataset.train_period.end="2020-12-31" \
    dataset.val_period.start="2021-01-01" \
    dataset.val_period.end="2021-01-31" \
    training.experiment_name="train_subset_1yr_1mo"
