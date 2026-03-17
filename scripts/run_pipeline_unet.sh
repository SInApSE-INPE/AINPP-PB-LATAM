#!/bin/bash
# ==============================================================================
# Script de Automação de Pipeline Completo para UNet (5 Épocas)
# Treinamento -> Inferência -> Avaliação (Métricas e Figuras)
# ==============================================================================

set -e

# Configuração do modelo e épocas
MODEL_PATH="unet/direct"
MODEL_CLEAN="unet_direct"
EPOCHS=5

echo "=========================================================="
echo "    INICIANDO PIPELINE AINPP PARA MODELO UNET (${EPOCHS} ÉPOCAS)  "
echo "=========================================================="

# Diretório base de saídas
BASE_DIR="$(pwd)/outputs/pipelines/${MODEL_CLEAN}"

# Overrides fixos (mantendo configuração leve para testes locais, mas com 5 épocas)
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
    echo "ERRO: Checkpoint '${CHECKPOINT_PATH}' não gerado ou não atingido."
    exit 1
fi

echo ""
echo "=========================================================="
echo ">> [2/3] INFERÊNCIA INDIVIDUAL: ${MODEL_PATH}"
echo "=========================================================="

# 2. Executa Inferência Single (Isolada, salva netcdf na pasta)
uv run python main.py task=infer $TRAIN_OVERRIDES inference.mode=single \
    +checkpoint="${CHECKPOINT_PATH}" \
    inference.output_dir="${BASE_DIR}/inference" \
    hydra.run.dir="${BASE_DIR}/infer_logs"

echo ""
echo "=========================================================="
echo ">> [3/3] AVALIAÇÃO CIENTÍFICA (BENCHMARK): ${MODEL_PATH}"
echo ">> Métricas e Geração Visual de Figuras"
echo "=========================================================="

# 3. Executa Avaliação e Visualização
uv run python main.py task=evaluate $TRAIN_OVERRIDES \
    +checkpoint="${CHECKPOINT_PATH}" \
    +evaluation.output_dir="${BASE_DIR}/metrics" \
    +visualization.output_dir="${BASE_DIR}/figures" \
    hydra.run.dir="${BASE_DIR}/eval_logs" \
    $FAST_OVERRIDES

echo ""
echo "=========================================================="
echo "    PIPELINE UNET FINALIZADO COM SUCESSO!                 "
echo "    Resultados (Métricas e Figuras) salvos em ${BASE_DIR}"
echo "=========================================================="
