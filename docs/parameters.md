# Referência de parâmetros

Esta página descreve todos os campos dos arquivos em `conf/`, os argumentos
adicionais dos datasets, modelos e perdas selecionáveis, e seu uso pelo `main.py`.
Os padrões abaixo são os do repositório; padrões de construtores Python são
identificados separadamente. Para comandos completos com o dataset WPR, consulte
[Workflow Examples](usage.md). Para assinaturas das funções internas, consulte a
[API](api.md).

## Como configurar

Execute os comandos na raiz, com o pacote instalado no ambiente ativo.
Hydra compõe `conf/config.yaml` e os grupos selecionados. `_target_` é o caminho
Python da classe que Hydra instancia; normalmente altere o grupo, não esse campo.
`${nome}` referencia outro campo da configuração, não uma variável Bash.

| Sintaxe | Uso | Exemplo |
| --- | --- | --- |
| `grupo=opção` | Selecionar um YAML | `model=unet/direct` |
| `chave=valor` | Alterar campo existente | `training.epochs=10` |
| `+chave=valor` | Adicionar campo ausente | `+dataset.dataset.seed=123` |
| `++chave=valor` | Adicionar ou substituir | `++dataset.overrides.test.group=test` |
| `null` | Valor nulo, sem aspas internas | `dataset.dataset.patch_height=null` |
| `true`, `false` | Booleanos | `model.bilinear=false` |
| `'chave=[a,b]'` | Lista; aspas protegem contra expansão do shell | `'model.features=[32,64,128,256]'` |
| `--cfg job --resolve` | Exibir configuração resolvida sem executar a tarefa | `python main.py training=default --cfg job --resolve` |
| `--help` | Listar grupos e configuração | `python main.py --help` |
| `--multirun` | Executar combinações de valores | `python main.py --multirun training=default training.lr=0.001,0.0003` |

Um multirun realmente inicia os treinamentos. Acrescente o caminho do Zarr e
use diretórios de checkpoint distintos para cada combinação.

```bash
python main.py task=train training=default model=unet/direct \
  dataset.dataset.zarr_path=/prj/cptec/nowcasting/data/benchmark/benchmark_dataset_wpr.zarr \
  training.lr=0.0003 dataset.train_loader.batch_size=2 \
  --cfg job --resolve
```

## Grupos e parâmetros globais

| Campo | Padrão | Descrição e uso |
| --- | --- | --- |
| `task` | `train` | `train`, `infer` ou `evaluate`. Não existe `task=visualize` no CLI unificado. |
| `checkpoint` | `null` | Arquivo de pesos para inferência/avaliação; deve corresponder à arquitetura. Não retoma treino. Arquivo ausente resulta em aviso e modelo com pesos novos. |
| `model` | `unet/direct` | Opções: `unet/direct`, `unet/autoregressive`, `convlstm/direct`, `afno/direct`, `resnet50/direct`, `inceptionv4/direct`, `xception/direct`. |
| `training` | `gan` | Perfis `default` e `gan`. Para treino supervisionado use explicitamente `training=default`. O CLI chama o motor supervisionado em ambos. |
| `dataset` | `gsmap` | Configura o dataset e os loaders; não seleciona automaticamente o arquivo WPR. |
| `loss` | `hybrid_mse_ssim` | Veja o catálogo de perdas abaixo. |
| `discriminator` | `patchgan` | Configuração disponível, mas não instanciada pelo treino do `main.py`. |
| `inference` | `default` | Perfil de inferência. |
| `evaluation` | `default` | Perfil de métricas. |
| `input_timesteps` | `12` | Quantidade de frames de entrada, inteiro positivo; com dados horários, 12 horas. |
| `output_timesteps` | `6` | Quantidade de frames previstos, inteiro positivo. Não configura a frequência temporal dos dados. |
| `input_channels` | `1` | Referenciado pelo YAML de ConvLSTM. Não muda automaticamente o loader nem todos os modelos. |
| `hidden_channels` | `[32,32,32]` | Canais das camadas ConvLSTM; uma entrada por camada. |
| `kernel_size` | `7` | Kernel ConvLSTM. UNet tem seu próprio `model.kernel_size=3`. |
| `system.num_workers` | `4` | Declarado, mas não aplicado pelos loaders de `main.py`. Use `dataset.*_loader.num_workers`. |
| `system.pin_memory` | `true` | Declarado, mas não aplicado pelo CLI. Use os campos dos loaders. |
| `system.sync_bn` | `false` | Não ativa conversão para SyncBatchNorm no CLI. |

O dataset atual retorna um canal por frame. Aumentar `input_channels` não empilha
variáveis: isso exige adaptação do carregamento. A UNet autoregressiva fixa 12/6
no YAML; altere também `model.input_timesteps` e `model.output_timesteps` quando
mudar os globais.

## Dataset: `dataset.dataset`

O prefixo duplo é intencional: `dataset` contém o objeto `dataset`, loaders e
sobrescritas por split. `_target_` aponta para
`ainpp_pb_latam.datasets.gsmap.AINPPPBLATAMDataset`.

| Campo após `dataset.dataset.` | Padrão | Efeito e restrições |
| --- | --- | --- |
| `zarr_path` | `/prj/ideeps/adriano.almeida/data/ainpp/legacy/AINPP-PB-LATAM.zarr` | Caminho local acessível ao Zarr; substituir pelo seu arquivo. |
| `input_timesteps` | `${input_timesteps}` | Comprimento da janela de entrada. |
| `output_timesteps` | `${output_timesteps}` | Comprimento do alvo imediatamente posterior à entrada. |
| `patch_height` | `320` | Altura em pixels; positiva e não maior que a grade. `null` usa toda a altura. |
| `patch_width` | `320` | Largura em pixels; `null` usa toda a largura. |
| `patch_stride_h` | `null` | Passo espacial vertical, inteiro positivo. `null` usa a altura do patch. |
| `patch_stride_w` | `null` | Passo horizontal; `null` usa a largura do patch. Passo menor cria sobreposição. |
| `consolidated` | `true` | Abre com metadados consolidados; use `false` se o store não os tiver. |
| `dtype` | `float32` | Tipo do tensor via `torch`, não conversão automática do modelo. Outros tipos precisam ser compatíveis com os pesos/operações. |

Os argumentos a seguir existem no construtor, mas não no YAML desse objeto.
Adicione-os com `+dataset.dataset.<campo>=...` quando necessário.

| Argumento | Padrão Python | Uso |
| --- | --- | --- |
| `group` | `train` | Grupo Zarr; sobrescritas por split têm precedência. |
| `stride` | `null` | Passo temporal em frames; se nulo, usa `output_timesteps` em `train` e a soma entrada+saída nos outros grupos. |
| `steps_per_epoch` | `null` | Inteiro positivo limita o número de amostras aleatórias. `null` percorre combinações de tempos e patches deterministicamente. |
| `seed` | `42` | Semente do gerador NumPy do dataset; não é uma semente global de PyTorch nem garante reprodutibilidade entre workers. |
| `input_var` | `gsmap_nrt` | Variável de entrada, com dimensões `time`, `lat`, `lon`. |
| `target_var` | `gsmap_mvk` | Variável alvo com a mesma geometria. |
| `return_metadata` | `false` | Retorna metadados como terceiro elemento; o treino supervisionado espera pares `(x,y)`, portanto mantenha `false` nesse fluxo. |

O loader substitui NaNs por zero. As janelas são baseadas em índices: ele não
valida continuidade horária dos timestamps. O número de frames precisa comportar
entrada+saída. Os patches finais podem se sobrepor na borda mesmo com passos
iguais ao tamanho do patch.

### Splits: `dataset.overrides`

| Campo completo | Padrão | Uso |
| --- | --- | --- |
| `dataset.overrides.train.group` | `train` | Grupo de treino. |
| `dataset.overrides.train.stride` | `1` | Passo dos inícios temporais elegíveis. |
| `dataset.overrides.train.steps_per_epoch` | `500` | 500 amostras aleatórias por época, não 500 batches. |
| `dataset.overrides.validation.group` | `validation` | Grupo de validação durante treino. |
| `dataset.overrides.validation.stride` | `6` | Passo temporal de validação. |
| `dataset.overrides.validation.steps_per_epoch` | `500` | Validação amostrada; `null` habilita percurso completo. |
| `dataset.overrides.test.group` | Ausente | Adicione `+dataset.overrides.test.group=test` para inferência/avaliação no teste. Sem isso, o construtor usa `train`. |
| `dataset.overrides.test.stride` | Ausente | Adicione para alterar o passo temporal de teste. |
| `dataset.overrides.test.steps_per_epoch` | Ausente | Adicione um inteiro para teste rápido aleatório; omitir preserva percurso determinístico. |

Exemplo de validação completa: `dataset.overrides.validation.steps_per_epoch=null`.
Com 500 amostras e batch 2, o loader tem 250 batches. Reduzir o passo aumenta a
quantidade de janelas candidatas; não altera as 500 amostras quando o limite existe.

### Loaders

| Campo | Padrão | Efeito |
| --- | --- | --- |
| `dataset.train_loader.batch_size` | `16` | Amostras por batch de treino; reduza para diminuir uso de memória. |
| `dataset.val_loader.batch_size` | `16` | Batch de validação, avaliação e inferência histórica. |
| `dataset.train_loader.num_workers` | `4` | Processos de carregamento de treino; `0` executa no processo principal. |
| `dataset.val_loader.num_workers` | `4` | Processos de carregamento dos demais fluxos com loader. |
| `dataset.train_loader.prefetch_factor` | `2` | Batches antecipados por worker; use `null` quando `num_workers=0`. |
| `dataset.train_loader.pin_memory` | `true` | Memória fixada para transferência ao dispositivo. |
| `dataset.val_loader.pin_memory` | `true` | Mesma opção nos outros loaders. |

Os dicionários são passados ao `DataLoader`; argumentos adicionais dependem da
API de PyTorch instalada. Evite adicionar `shuffle` ao loader de validação: o CLI
já passa `shuffle=False` explicitamente em avaliação/inferência histórica.

## Treinamento: `training`

| Campo | `default` / `gan` | Efeito no CLI atual |
| --- | --- | --- |
| `mode` | `supervised` / `gan` | Rótulo; não seleciona o motor em `main.py`. |
| `epochs` | `50` / `100` | Máximo de épocas; inteiro positivo. |
| `lr` | `0.001` / ausente | Taxa de Adam, tem prioridade sobre `lr_g` se presente. |
| `lr_g` | ausente / `0.0002` | Fallback para a taxa do único otimizador do CLI. |
| `lr_d` | ausente / `0.0002` | Não utilizado pelo CLI supervisionado. |
| `beta1` | fallback `0.9` / `0.5` | Primeiro coeficiente de Adam. Use `+training.beta1=...` no perfil `default`. |
| `beta2` | fallback `0.999` / `0.999` | Segundo coeficiente de Adam. |
| `lambda_pixel` | ausente / `100.0` | Peso do conteúdo no motor GAN separado; não utilizado pelo CLI. |
| `batch_size` | `16` / ausente | Não define batches dos loaders. |
| `scheduler.patience` | `5` / ausente | Declarado; o motor não cria scheduler. |
| `scheduler.factor` | `0.1` / ausente | Declarado; não reduz automaticamente a taxa. |

Adam é fixo em `build_optimizer`; não há seletor de otimizador no YAML.

### Checkpoints e parada antecipada

Os dois perfis compartilham estes valores.

| Campo após `training.` | Padrão | Efeito real |
| --- | --- | --- |
| `checkpoint.enabled` | `true` | Habilita checkpoints periódicos; não desliga o salvamento por early stopping. |
| `checkpoint.dir` | `outputs/<data>/<hora>/early_stopping` | Diretório de todos os checkpoints desse motor. |
| `checkpoint.interval` | `5` | Salva a cada N épocas; use inteiro maior que zero. |
| `checkpoint.save_best` | `true` | Declarado, mas não consultado pelo motor. |
| `early_stopping.enabled` | `true` | Habilita parada antecipada e salvamento do melhor modelo. |
| `early_stopping.patience` | `10` | Épocas sem melhoria suficiente antes de parar. |
| `early_stopping.delta` | `0.001` | Redução mínima de loss de validação considerada melhoria. |
| `early_stopping.mode` | `min` | O motor não repassa esse campo; minimiza a loss mesmo que se configure `max`. |

Melhor modelo: `best_model.pt`. Periódicos:
`checkpoint_model_epoch_005.pt`, etc. São `state_dict`s, sem estado de Adam ou
contador de época. Desabilitar early stopping também impede a gravação do melhor
modelo por esse mecanismo. Imagens de exemplo são gravadas em `samples/`, caminho
fixo no motor.

## Modelos: `model`

Todos precisam respeitar tensores `(B,T,C,H,W)`. Alterar arquitetura exige usar
os mesmos argumentos ao carregar o checkpoint.

### UNet direta e autoregressiva

| Campo após `model.` | Padrão YAML | Uso |
| --- | --- | --- |
| `input_timesteps` | Global na direta; `12` na autoregressiva | Frames de contexto. |
| `output_timesteps` | Global na direta; `6` na autoregressiva | Horizontes previstos. |
| `input_channels` | `1` | Canais por frame. |
| `output_channels` | `1`, somente direta | Canais de saída; a autoregressiva usa os canais de entrada. |
| `features` | `[64,128,256,512]` | Larguras dos níveis; valores maiores aumentam capacidade e memória. |
| `kernel_size` | `3` | Tamanho do kernel; use ímpar positivo para a geometria esperada. |
| `bilinear` | `true` | Upsampling bilinear; `false` usa convolução transposta. |
| `nonnegativity` | `relu` | `relu` corta negativos, `softplus` gera valores positivos suavemente, `none` deixa saída sem restrição. |

```bash
python main.py training=default model=unet/direct \
  'model.features=[32,64,128,256]' model.nonnegativity=softplus --cfg job --resolve
```

### ConvLSTM

| Campo após `model.` | Padrão composto | Uso |
| --- | --- | --- |
| `input_channels` | `${input_channels}` = `1` | Canais por frame. |
| `hidden_channels` | `${hidden_channels}` = `[32,32,32]` | Larguras e quantidade de camadas recorrentes. O construtor isolado usa `[64,64,64]`. |
| `kernel_size` | `${kernel_size}` = `7` | Kernel espacial recorrente. |
| `output_timesteps` | `${output_timesteps}` = `6` | Passos previstos pelo decoder. |

Não há argumento `model.input_timesteps` nesse construtor: o contexto vem do tensor.

### AFNO

Os campos temporais existem no YAML; os demais precisam de `+model.<campo>=...`.

| Campo | Padrão | Uso/restrição |
| --- | --- | --- |
| `input_timesteps` | Global `12` (Python: `6`) | Frames empilhados na entrada. |
| `output_timesteps` | Global `6` | Frames de saída. |
| `img_size` | Python `[880,970]` | Deve coincidir com altura/largura do dado entregue; para patches padrão use `[320,320]`. |
| `input_channels` | Python `1` | Canais por frame de entrada. |
| `output_channels` | Python `1` | Canais por frame previsto. |
| `embed_dim` | Python `256` | Dimensão do embedding, divisível por `num_blocks`. |
| `depth` | Python `8` | Número de blocos AFNO. |
| `patch_size` | Python `10` | Patch interno da rede; deve dividir ambas as dimensões de `img_size`. Não é o recorte do dataset. |
| `num_blocks` | Python `8` | Partições de canais da operação espectral. |

Exemplo: `model=afno/direct '+model.img_size=[320,320]' +model.depth=4`.

### ResNet50, InceptionV4 e Xception

Todos expõem `model.input_timesteps=${input_timesteps}` e
`model.output_timesteps=${output_timesteps}`. O argumento adicional
`pretrained` tem padrão Python `true`: carrega pesos do encoder, podendo precisar
de download/cache. Use `+model.pretrained=false` para inicializar sem esses pesos.
Esses construtores não expõem `input_channels`; o fluxo pressupõe um canal por
frame. Exemplo: `model=resnet50/direct +model.pretrained=false`.

## Perdas: `loss`

Os limiares têm a unidade dos alvos (mm/h para o GSMaP horário sem transformação).
Não convertem unidades ou normalizam dados automaticamente.

| Grupo | Campos e padrões YAML | Significado |
| --- | --- | --- |
| `weighted_mse` | `alpha=5.0`, `threshold=0.1` | Acima do limiar, peso `1 + alpha * target`; abaixo, peso 1. `alpha<=0` desativa ponderação. |
| `logcosh` | Sem argumentos | Média de `log(cosh(pred-target))`. |
| `huber` | Sem campos adicionais | Transição quadrática/linear; argumento Python `delta=1.0`, alterável com `+loss.delta=2.0`. |
| `dice` | `threshold=0.5`, `smooth=1e-6` | Alvo binário acima do limiar; previsão suavizada por sigmoid. `smooth` estabiliza a divisão. |
| `focal` | `alpha=0.25`, `gamma=2.0`, `threshold=0.1` | Multiplicador da loss, foco nos exemplos difíceis e limiar do alvo. A implementação multiplica todo o termo por `alpha`, não usa pesos distintos por classe. |
| `torrential` | `thresholds=[5,20,50]`, `weights=[2,5,10]` | MSE com pesos por faixa; o último limiar atingido determina o peso. Listas com mesmo tamanho e limiares em ordem crescente. |
| `spectral` | `alpha=1.0`, `beta=1.0` | Pesos do erro de amplitude FFT e erro no plano complexo, respectivamente. |
| `hybrid_mse_ssim` | `weights=[1.0,0.2]`, `losses` com 2 componentes | Soma ponderada de WeightedMSE (`alpha=2`, `threshold=0`) e SSIM (`window_size=11`, `in_channels=1`). |
| `sota` | `weights=[1.0,0.1,0.05]`, `losses` com 3 componentes | Torrential (`thresholds=[10]`, `weights=[5]`), Spectral (`alpha=1`, `beta=0.5`) e Perceptual (`layer_ids=[3,8,15]`). |

Perdas híbridas exigem um peso por componente. Para editar um item use índice:

```bash
python main.py training=default loss=hybrid_mse_ssim \
  'loss.weights=[1.0,0.3]' loss.losses.0.alpha=3.0 \
  loss.losses.1.window_size=7 --cfg job --resolve
```

`window_size` é a janela espacial SSIM (ímpar positivo); `in_channels` deve
corresponder aos dados. `layer_ids` seleciona camadas de features VGG16. A perda
perceptual tenta carregar pesos pré-treinados e usa MSE como fallback se falhar.
O padrão Python de `layer_ids` é `[3,8,15,22]`, diferente do grupo `sota`.

`BinaryFocalLoss` espera logits; trocar a loss de uma rede com saída ReLU não
transforma o pipeline em classificação adequada. `CrossEntropyLoss` está disponível
na API, sem grupo YAML: seu argumento `weights=None` aceita pesos por classe e
requer logits/classes inteiras, incompatíveis diretamente com os alvos de regressão
atuais. As perdas SSIM e Perceptual isoladas também não têm grupo YAML próprio.

## Discriminador: `discriminator`

`patchgan` aponta para `PatchDiscriminator3D`. Estes campos só têm efeito quando
um fluxo instancia e utiliza esse discriminador; `main.py` não o faz.

| Campo | Padrão | Uso |
| --- | --- | --- |
| `input_channels` | `1` | Canais do tensor `(B,C,T,H,W)`. O motor GAN concatena histórico e futuro no tempo, não nos canais. |
| `ndf` | `64` | Número base de filtros. |
| `n_layers` | `1` | Profundidade do discriminador. |
| `norm_type` | `instance` | `batch` usa BatchNorm3d; `instance` usa InstanceNorm3d (também é o fallback para outros valores). |

## Inferência: `inference`

| Campo | Padrão | Efeito |
| --- | --- | --- |
| `mode` | `historical` | `single` prevê a primeira amostra; `historical` percorre o loader. |
| `output_dir` | `outputs/inference` | Base interpolada nos caminhos específicos. |
| `batch_size` | `16` | Primeiro eixo do chunk Zarr; não controla o batch do loader. |
| `historical.output_format` | `zarr` | Declarado; a implementação grava Zarr independentemente desse valor. |
| `historical.zarr_store` | `${inference.output_dir}/predictions.zarr` | Destino do store, sobrescrito na primeira escrita. |
| `single.output_format` | `nc` | `nc` grava NetCDF; use `pt` para tensor PyTorch. Outros valores caem no fallback `.pt`. |
| `single.output_dir` | `${inference.output_dir}/single` | Base da hierarquia ano/mês/dia. |

Use `dataset.val_loader.batch_size` para memória/batches da inferência histórica.
A previsão individual tem timestamp fixo no CLI (`20260316_1200`); não seleciona
uma data por argumento. Os arquivos atuais não preservam timestamps/coordenadas
reais dos patches. O store histórico contém `predictions` com eixos
`(amostra,horizonte,canal,altura,largura)`, sem reconstrução de mosaico.

## Avaliação: `evaluation`

| Campo | Padrão | Efeito |
| --- | --- | --- |
| `region` | `ainpp-amazon-basin` | Declarado; não recorta o dataset nem filtra região no avaliador. |
| `checkpoint` | String vazia | Não carrega pesos; use `checkpoint` na raiz. |
| `thresholds_mm_h` | `[0.1,1.0,5.0,10.0]` | Limiares usados nos cálculos dependentes de eventos. |
| `lead_times_min` | `[10,20,30,40,50,60]` | Lido, mas não aplicado aos rótulos: os resultados usam `T+1`, `T+2`, etc. Não reamostra o dado. |
| `categorical` | `true` | Ativa métricas de ocorrência de eventos. |
| `continuous` | `true` | Ativa erros e associação entre valores contínuos. |
| `probabilistic` | `true` | Ativa o ramo probabilístico implementado; não transforma o modelo em ensemble. |
| `object_based` | `true` | Ativa métricas de objetos/eventos espaciais. |
| `sharpness` | `true` | Ativa métricas de estrutura/nitidez. |
| `consistency` | `true` | Ativa métricas de consistência das distribuições. |
| `max_batches` | `null` | Não aplicado pelo loop atual. |
| `output_dir` | Ausente; fallback `outputs/evaluation` | Adicione `+evaluation.output_dir=...`; destino de `evaluation_summary.csv` e Parquet opcional. |

Para dados horários, registre `'evaluation.lead_times_min=[60,120,180,240,300,360]'`
na configuração do experimento, mas interprete `T+1` como primeiro frame previsto:
a conversão de rótulos para minutos ainda não foi implementada. Para teste rápido,
use `+dataset.overrides.test.steps_per_epoch=8` (amostragem aleatória), não
`evaluation.max_batches`. Parquet exige `pyarrow` ou `fastparquet`.

## Visualização: `visualization`

O grupo `visualization/default.yaml` não está nos `defaults` da raiz.
`+visualization=default` o inclui, mas o avaliador do CLI só usa
`visualization.output_dir`; ele não repassa o perfil de estilo ao gerador de figuras.

| Campo | Padrão do perfil | Uso |
| --- | --- | --- |
| `output_dir` | Ausente; CLI usa `outputs/figures` | `+visualization.output_dir=...` define destino das figuras da avaliação. |
| `style.context` | `paper` | Contexto Seaborn no `VisualizationRunner`/`set_style`. |
| `style.style` | `whitegrid` | Estilo Seaborn. |
| `style.palette` | `deep` | Paleta Seaborn. |
| `style.font_family` | `sans-serif` | Família de fonte Matplotlib. |
| `style.dpi` | `300` | Resolução da figura. |
| `maps.cmap` | `viridis` | Mapa de cores de precipitação no runner. |
| `maps.diff_cmap` | `coolwarm` | Mapa de cores de diferenças no runner. |
| `animation.fps` | `5` | Frames por segundo das animações do runner. |

Esses campos de estilo são para a API/runner separado, que espera `metrics.json`
e/ou `sample_*.npz`. Não assuma que os CSVs da avaliação alimentam esse runner.

## Hydra, caminhos e ambiente

| Campo/configuração | Padrão do projeto | Uso |
| --- | --- | --- |
| `hydra.run.dir` | `outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}` | Diretório de execução Hydra para uma rodada. |
| `hydra.sweep.dir` | `multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}` | Base para multirun. |
| `hydra.sweep.subdir` | `${hydra.job.num}` | Subdiretório por combinação. |
| `hydra.job.chdir` | Não definido no projeto | Evite habilitar sem revisar caminhos relativos de dados/checkpoints. |
| `CUDA_VISIBLE_DEVICES` | Ambiente externo | Seleciona GPUs visíveis, por exemplo `CUDA_VISIBLE_DEVICES=0 python main.py ...`. |
| `HYDRA_FULL_ERROR` | Ambiente externo | `HYDRA_FULL_ERROR=1` exibe traceback completo. |

`hydra.run.dir` não redireciona automaticamente todos os artefatos. Configure
`training.checkpoint.dir`, `inference.output_dir`, `evaluation.output_dir` e
`visualization.output_dir` conforme o fluxo. Os logs/configurações Hydra permitem
registrar os overrides, mas não garantem determinismo científico.

Não há parâmetros ativos no CLI para AMP, gradient accumulation, seleção de
otimizador, retomada completa de treino ou inicialização DDP. Use os campos
realmente consumidos acima; adicionar uma chave arbitrária com `+` não implementa
a funcionalidade correspondente.
