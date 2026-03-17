#!/bin/bash
# Script para testar rapidamente o treinamento com todos os modelos suportados (smoke test).
# A intenção é validar se o pipeline roda sem quebrar (overriddings curtos).

set -e

# Overrides padrão para um teste super rápido (1 época, batches mínimos)
FAST_OVERRIDES="training.epochs=1 dataset.train_loader.batch_size=2 dataset.val_loader.batch_size=2 dataset.overrides.train.steps_per_epoch=2 dataset.overrides.validation.steps_per_epoch=2 system.num_workers=0"

echo "==================================================="
echo "    Iniciando Teste Rápido de Todos os Modelos     "
echo "==================================================="

# Modelos padrão e a configuração equivalente no Hydra (conf/model/*)
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
    echo ">> Testando Arquitetura: $model (Padrão Supervisionado)"
    uv run python main.py task=train model=$model training=default ~discriminator $FAST_OVERRIDES $EXTRA_FLAGS
done

# Para testar a GAN, o discriminator precisa estar ativado e o treino como gan
echo ">> Testando Arquitetura: unet/direct (Adversarial/GAN)"
uv run python main.py task=train model=unet/direct training=gan discriminator=patchgan $FAST_OVERRIDES

echo "==================================================="
echo "         Todos os testes foram concluídos!         "
echo "==================================================="
