#!/bin/bash
# Script to quickly test training with all supported models (smoke test).
# Intention is to validate if pipeline runs without breaking (short overrides).

set -e

# Default overrides for a super fast test (1 epoch, minimal batches)
FAST_OVERRIDES="training.epochs=1 dataset.train_loader.batch_size=2 dataset.val_loader.batch_size=2 dataset.overrides.train.steps_per_epoch=2 dataset.overrides.validation.steps_per_epoch=2 system.num_workers=0"

echo "==================================================="
echo "    Starting Fast Test for All Models     "
echo "==================================================="

# Default models and the equivalent configuration in Hydra (conf/model/*)
models=(
    "afno/direct"
    "convlstm/direct"
    "inceptionv4/direct"
    "resnet50/direct"
    "unet/direct"
    "unet/autoregressive"
    "xception/direct"
)

for model in "${models[@]}"; do
    EXTRA_FLAGS=""
    if [[ "$model" == *"afno"* ]]; then
        EXTRA_FLAGS="+model.img_size=[320,320]"
    fi
    echo ">> Testing Architecture: $model (Supervised Standard)"
    uv run python main.py task=train model=$model training=default ~discriminator $FAST_OVERRIDES $EXTRA_FLAGS
done

# To test GAN, discriminator needs to be activated and training as gan
echo ">> Testing Architecture: unet/direct (Adversarial/GAN)"
uv run python main.py task=train model=unet/direct training=gan discriminator=patchgan $FAST_OVERRIDES

echo "==================================================="
echo "         All tests have been completed!         "
echo "==================================================="
