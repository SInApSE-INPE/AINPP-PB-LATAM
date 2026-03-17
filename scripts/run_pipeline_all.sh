#!/bin/bash
# ==============================================================================
# Script de Automação de Pipeline Completo: Treinamento -> Inferência -> Avaliação
# Para todos os modelos implementados na biblioteca AINPP-PB-LATAM
# ==============================================================================

set -e

# Modelos padrão e a configuração equivalente no Hydra
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

# Overrides fixos para forçar 5 épocas e processamento leve (para ambiente local se necessário)
# Se for rodar numa máquina grande do Santos Dumont, remover os limits de batch.
EPOCHS=2
FAST_OVERRIDES="training.epochs=${EPOCHS} dataset.train_loader.batch_size=2 dataset.val_loader.batch_size=2 +dataset.overrides.test.steps_per_epoch=2 +dataset.overrides.test.group=test"

echo "=========================================================="
echo "    INICIANDO PIPELINES DE MODELOS AINPP (5 ÉPOCAS)       "
echo "=========================================================="

for model_path in "${models[@]}"; do
    MODEL_CLEAN=$(echo $model_path | tr '/' '_')
    BASE_DIR=$(pwd)/outputs/pipelines/${MODEL_CLEAN}
    
    echo ""
    echo "=========================================================="
    echo ">> [1/3] TREINAMENTO: ${model_path}"
    echo ">> Destino: ${BASE_DIR}"
    echo "=========================================================="
    
    # Tratativas extras baseadas na arquitetura do modelo
    EXTRA_FLAGS=""
    if [[ "$model_path" == *"afno"* ]]; then
        # Correção de dimensionalidade exigida pelo ViT/AFNO
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
        echo "ERRO: Checkpoint '${CHECKPOINT_PATH}' não gerado ou não atingido."
        # Pode continuar proximo modelo (continue) ou desabar o script
        continue
    fi
    
    echo ""
    echo "=========================================================="
    echo ">> [2/3] INFERÊNCIA INDIVIDUAL: ${model_path}"
    echo "=========================================================="
    
    # 2. Executa Inferência Single (Isolada, salva netcdf na pasta)
    uv run python main.py task=infer $TRAIN_OVERRIDES inference.mode=single \
        +checkpoint="${CHECKPOINT_PATH}" \
        inference.output_dir="${BASE_DIR}/inference" \
        hydra.run.dir="${BASE_DIR}/infer_logs" \
        $EXTRA_FLAGS
        
    echo ""
    echo "=========================================================="
    echo ">> [3/3] AVALIAÇÃO CIENTÍFICA (BENCHMARK): ${model_path}"
    echo "=========================================================="
    
    uv run python main.py task=evaluate $TRAIN_OVERRIDES \
        +checkpoint="${CHECKPOINT_PATH}" \
        +evaluation.output_dir="${BASE_DIR}/metrics" \
        +visualization.output_dir="${BASE_DIR}/figures" \
        hydra.run.dir="${BASE_DIR}/eval_logs" \
        $FAST_OVERRIDES $EXTRA_FLAGS
        
    echo "[CONCLUÍDO] Pipeline de ${MODEL_CLEAN} gerou todos os artefatos com sucesso!"
done

echo "=========================================================="
echo "    TODOS OS MODELOS FINALIZADOS. RESULTADOS SALVOS.      "
echo "=========================================================="
