#!/bin/bash
# ==============================================================================
# Complete Pipeline Automation Script: Training -> Inference -> Evaluation
# For all models implemented in the library AINPP-PB-LATAM
# ==============================================================================

set -e

# Default models and the equivalent configuration in Hydra
models=(
    "afno/direct"
    "convlstm/direct"
    "gan/direct"
    "inceptionv4/direct"
    "resnet50/direct"
    "unet/direct"
    "unet/autoregressive"
    "xception/direct"
)

# Fixed overrides to force 5 epochs and light processing (for local environment if necessary)
# If running in a large machine at Santos Dumont, remove batch limits.
EPOCHS=2
FAST_OVERRIDES="training.epochs=${EPOCHS} dataset.train_loader.batch_size=2 dataset.val_loader.batch_size=2 +dataset.overrides.test.steps_per_epoch=2 +dataset.overrides.test.group=test"

echo "=========================================================="
echo "    STARTING AINPP MODEL PIPELINES (5 EPOCHS)       "
echo "=========================================================="

for model_path in "${models[@]}"; do
    MODEL_CLEAN=$(echo $model_path | tr '/' '_')
    BASE_DIR=$(pwd)/outputs/pipelines/${MODEL_CLEAN}
    
    echo ""
    echo "=========================================================="
    echo ">> [1/3] TREINAMENTO: ${model_path}"
    echo ">> Destino: ${BASE_DIR}"
    echo "=========================================================="
    
    # Extra treatments based on model architecture
    EXTRA_FLAGS=""
    if [[ "$model_path" == *"afno"* ]]; then
        # Dimensionality correction required by ViT/AFNO
        EXTRA_FLAGS="+model.img_size=[320,320]"
    fi
    
    TRAIN_OVERRIDES="model=$model_path training=default ~discriminator"
    if [[ "$model_path" == *"gan"* ]]; then
        # GAN exige discriminador e pipeline adversarial
        TRAIN_OVERRIDES="model=unet/direct training=gan discriminator=patchgan"
    fi

    # 1. Executa Treinamento
    uv run python main.py task=train $TRAIN_OVERRIDES $FAST_OVERRIDES \
        training.checkpoint.dir="${BASE_DIR}/checkpoints" \
        hydra.run.dir="${BASE_DIR}/train_logs" \
        $EXTRA_FLAGS

    CHECKPOINT_PATH="${BASE_DIR}/checkpoints/best_model.pt"
    
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "ERROR: Checkpoint '${CHECKPOINT_PATH}' not generated or not reached."
        # Can continue to next model (continue) or fall script
        continue
    fi
    
    echo ""
    echo "=========================================================="
    echo ">> [2/3] INDIVIDUAL INFERENCE: ${model_path}"
    echo "=========================================================="
    
    # 2. Executes Single Inference (Isolated, saves netcdf in folder)
    uv run python main.py task=infer $TRAIN_OVERRIDES inference.mode=single \
        +checkpoint="${CHECKPOINT_PATH}" \
        inference.output_dir="${BASE_DIR}/inference" \
        hydra.run.dir="${BASE_DIR}/infer_logs" \
        $EXTRA_FLAGS
        
    echo ""
    echo "=========================================================="
    echo ">> [3/3] SCIENTIFIC EVALUATION (BENCHMARK): ${model_path}"
    echo "=========================================================="
    
    uv run python main.py task=evaluate $TRAIN_OVERRIDES \
        +checkpoint="${CHECKPOINT_PATH}" \
        +evaluation.output_dir="${BASE_DIR}/metrics" \
        +visualization.output_dir="${BASE_DIR}/figures" \
        hydra.run.dir="${BASE_DIR}/eval_logs" \
        $FAST_OVERRIDES $EXTRA_FLAGS
        
    echo "[COMPLETED] Pipeline de ${MODEL_CLEAN} successfully generated all artifacts!"
done

echo "=========================================================="
echo "    TODOS OS MODELOS FINALIZADOS. RESULTADOS SALVOS.      "
echo "=========================================================="
