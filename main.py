import logging
from pathlib import Path
from typing import Callable

import hydra
from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path, instantiate
from omegaconf import DictConfig, OmegaConf

# Require proper installation instead of sys.path hacks.
# Ensure you have run: uv pip install -e .



from ainpp.datasets import NowcastingDataset
from ainpp.models import unet  # noqa: F401  # Ensure model modules are discoverable
from ainpp.losses import HybridLoss  # noqa: F401
from ainpp.engine import run_training
from ainpp.utils import build_optimizer, build_loss
from ainpp.visualization.samples import save_epoch_sample  # noqa: F401
from ainpp.evaluation.evaluator import Evaluator
from ainpp.inference import Inferencer

LOG = logging.getLogger("ainpp.cli")



def _build_dataloader(
    dataset_cfg: DictConfig,
    loader_cfg: DictConfig,
    overrides: DictConfig,
    split: str,
):
    import torch
    from torch.utils.data import DataLoader

    ds_kwargs = overrides.get(split, {}) if overrides else {}
    dataset = instantiate(dataset_cfg, **ds_kwargs)
    return DataLoader(dataset=dataset, **loader_cfg)


def _run_train(cfg: DictConfig) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DistributedSampler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader = _build_dataloader(
        cfg.dataset.dataset, cfg.dataset.train_loader, cfg.dataset.overrides, split="train"
    )
    val_loader = _build_dataloader(
        cfg.dataset.dataset, cfg.dataset.val_loader, cfg.dataset.overrides, split="validation"
    )

    # Model & criterion
    model = instantiate(cfg.model).to(device)
    criterion: nn.Module = build_loss(cfg.loss).to(device)

    # Optimizer
    optimizer = build_optimizer(model.parameters(), cfg.training)

    # Distributed sampler (optional)
    train_sampler = (
        train_loader.sampler
        if isinstance(train_loader.sampler, DistributedSampler)
        else None
    )

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=cfg.training.epochs,
        criterion=criterion,
        early_stopping=cfg.training.early_stopping,
        checkpoint=cfg.training.checkpoint,
        train_sampler=train_sampler,
    )


def _run_evaluate(cfg: DictConfig) -> None:
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_kwargs = cfg.dataset.overrides.get("test", {}) if cfg.dataset.get("overrides") else {}
    test_dataset = instantiate(cfg.dataset.dataset, **ds_kwargs)
    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        **cfg.dataset.val_loader,
    )

    model = instantiate(cfg.model).to(device)
    if cfg.get("checkpoint"):
        checkpoint_path = Path(cfg.checkpoint)
        if checkpoint_path.exists():
            LOG.info("Loading checkpoint from %s", checkpoint_path)
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
        else:
            LOG.warning("Checkpoint not found at %s; evaluating fresh model", checkpoint_path)

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        config=cfg,
        standardizer=None,
        device=device,
    )
    df_summary = evaluator.evaluate()
    
    # Generate visualization figures Based on the aggregated dataframe
    vis_dir = cfg.get("visualization", {}).get("output_dir", "outputs/figures")
    from ainpp.visualization.generate_figures import generate_benchmark_figures
    generate_benchmark_figures(df_summary, vis_dir)


def _run_infer(cfg: DictConfig) -> None:
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.model).to(device)
    
    if cfg.get("checkpoint"):
        checkpoint_path = Path(cfg.checkpoint)
        if checkpoint_path.exists():
            LOG.info("Loading checkpoint for Inference from %s", checkpoint_path)
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
        else:
            LOG.warning("Checkpoint not found at %s; predicting with fresh model", checkpoint_path)
    
    inferencer = Inferencer(model=model, config=cfg, device=device)
    mode = cfg.inference.mode

    if mode == "historical":
        ds_kwargs = cfg.dataset.overrides.get("test", {}) if cfg.dataset.get("overrides") else {}
        test_dataset = instantiate(cfg.dataset.dataset, **ds_kwargs)
        test_loader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            **cfg.dataset.val_loader, # Usa os args de validation (batch_size, num_workers)
        )
        inferencer.infer_historical(dataloader=test_loader)
    
    elif mode == "single":
        # Simula carregamento do primeiro batch e pega a primeira amostra
        ds_kwargs = cfg.dataset.overrides.get("test", {}) if cfg.dataset.get("overrides") else {}
        test_dataset = instantiate(cfg.dataset.dataset, **ds_kwargs)
        
        # Pega fisicamente o primeiro shape compatível (depende de como o Dataset retorna)
        sample = test_dataset[0]
        if isinstance(sample, (list, tuple)):
            input_tensor = sample[0]
        else:
            input_tensor = sample
            
        base_timestamp = "20260316_1200" # Exemplo hardcoded temporariamente para compatibilidade CLI
        inferencer.infer_single(input_tensor=input_tensor, base_timestamp=base_timestamp)
    else:
        raise ValueError(f"Inference mode {mode} not supported. Use 'historical' or 'single'.")


TASK_HANDLERS: dict[str, Callable[[DictConfig], None]] = {
    "train": _run_train,
    "evaluate": _run_evaluate,
    "infer": _run_infer,
}


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    task = cfg.get("task", "train")
    if task not in TASK_HANDLERS:
        valid = ", ".join(TASK_HANDLERS)
        raise ValueError(f"Unsupported task '{task}'. Choose one of: {valid}")

    LOG.info("Running task: %s", task)
    handler = TASK_HANDLERS[task]
    handler(cfg)


if __name__ == "__main__":
    main()
