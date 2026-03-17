#!/bin/bash
# ==============================================================================
# Complete Pipeline Automation Script for UNet (5 Epochs)
# Training -> Inference -> Evaluation (Metrics and Figures)
# ==============================================================================

set -e

# Model configuration and epochs
MODEL_PATH="unet/direct"
MODEL_CLEAN="unet_direct"
EPOCHS=5

echo "=========================================================="
echo "    STARTING AINPP PIPELINE FOR UNET MODEL (${EPOCHS} EPOCHS)  "
echo "=========================================================="

# Base outputs directory
BASE_DIR="$(pwd)/outputs/pipelines/${MODEL_CLEAN}"

# Fixed overrides (keeping light configuration for local tests, but with 5 epochs)
FAST_OVERRIDES="training.epochs=${EPOCHS} dataset.train_loader.batch_size=2 dataset.val_loader.batch_size=2 +dataset.overrides.test.steps_per_epoch=2 +dataset.overrides.test.group=test"
TRAIN_OVERRIDES="model=${MODEL_PATH} training=default ~discriminator"

echo ""
echo "=========================================================="
echo ">> [1/3] TREINAMENTO: ${MODEL_PATH}"
echo ">> Destino: ${BASE_DIR}"
echo "=========================================================="

# 1. Executa Treinamento
uv run python main.py task=train $TRAIN_OVERRIDES $FAST_OVERRIDES \
    training.checkpoint.dir="${BASE_DIR}/checkpoints" \
    hydra.run.dir="${BASE_DIR}/train_logs"

CHECKPOINT_PATH="${BASE_DIR}/checkpoints/best_model.pt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint '${CHECKPOINT_PATH}' not generated or not reached."
    exit 1
fi

echo ""
echo "=========================================================="
echo ">> [2/3] INDIVIDUAL INFERENCE: ${MODEL_PATH}"
echo "=========================================================="

# 2. Executes Single Inference (Isolated, saves netcdf in folder)
uv run python main.py task=infer $TRAIN_OVERRIDES inference.mode=single \
    +checkpoint="${CHECKPOINT_PATH}" \
    inference.output_dir="${BASE_DIR}/inference" \
    hydra.run.dir="${BASE_DIR}/infer_logs"

echo ""
echo "=========================================================="
echo ">> [3/3] SCIENTIFIC EVALUATION (BENCHMARK): ${MODEL_PATH}"
echo ">> Metrics and Visual Generation of Figures"
echo "=========================================================="

# 3. Executes Evaluation and Visualization
uv run python main.py task=evaluate $TRAIN_OVERRIDES \
    +checkpoint="${CHECKPOINT_PATH}" \
    +evaluation.output_dir="${BASE_DIR}/metrics" \
    +visualization.output_dir="${BASE_DIR}/figures" \
    hydra.run.dir="${BASE_DIR}/eval_logs" \
    $FAST_OVERRIDES

echo ""
echo "=========================================================="
echo "    PIPELINE UNET FINALIZADO COM SUCESSO!                 "
echo "    Results (Metrics and Figures) saved in ${BASE_DIR}"
echo "=========================================================="
