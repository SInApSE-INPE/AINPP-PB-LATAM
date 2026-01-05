from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import GSMaPZarrDataset
from models.unet import UNetMultiHorizon


logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -----------------------------
# Distributed utilities
# -----------------------------
def init_distributed() -> Tuple[int, int, int]:
    """
    Initialize DDP from torchrun environment variables.

    Returns
    -------
    local_rank, global_rank, world_size
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def cleanup_distributed() -> None:
    """Destroy the DDP process group."""
    dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def rank0_print(rank: int, msg: str) -> None:
    if is_rank0(rank):
        print(msg)


def ddp_mean(value: float, device: torch.device) -> float:
    """All-reduce mean for a scalar float across all ranks."""
    t = torch.tensor(value, device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t.item())


# -----------------------------
# Losses (Factory)
# -----------------------------
class WeightedMSELoss(nn.Module):
    """
    Weighted MSE emphasizing high-intensity targets.

    weight = high_weight if target > threshold else 1.0
    """

    def __init__(self, threshold: float = 0.5, high_weight: float = 5.0) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.high_weight = float(high_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = (pred - target) ** 2
        weights = torch.where(target > self.threshold, self.high_weight, 1.0).to(pred.dtype)
        return (mse * weights).mean()


class SharpnessLoss(nn.Module):
    """
    Composite loss intended to improve sharpness/structure:
      - weighted MSE (focus on peaks)
      - L1 (overall sharpness)
      - 1 - SSIM (structural similarity)

    Notes
    -----
    SSIM expects a correct data_range. If your tensors are normalized, it can be stable.
    If targets are standardized (mean/std), SSIM interpretation is less physical.
    """

    def __init__(
        self,
        w_mse: float = 1.0,
        w_l1: float = 0.5,
        w_ssim: float = 0.1,
        threshold: float = 0.5,
        high_weight: float = 5.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.w_mse = float(w_mse)
        self.w_l1 = float(w_l1)
        self.w_ssim = float(w_ssim)
        self.threshold = float(threshold)
        self.high_weight = float(high_weight)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = (pred - target) ** 2
        weights = torch.where(target > self.threshold, self.high_weight, 1.0).to(pred.dtype)
        loss_mse = (mse * weights).mean()

        loss_l1 = F.l1_loss(pred, target)

        # Avoid degenerate data_range
        data_range = (target.max() - target.min()).clamp_min(self.eps)
        loss_ssim = 1.0 - ssim(pred, target, data_range=float(data_range.item()), size_average=True)

        return self.w_mse * loss_mse + self.w_l1 * loss_l1 + self.w_ssim * loss_ssim


def build_loss(args: argparse.Namespace) -> nn.Module:
    """Factory: create a loss module from CLI args."""
    name = args.loss.lower()

    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "weighted_mse":
        return WeightedMSELoss(threshold=args.loss_threshold, high_weight=args.loss_high_weight)
    if name == "sharpness":
        return SharpnessLoss(
            w_mse=args.w_mse,
            w_l1=args.w_l1,
            w_ssim=args.w_ssim,
            threshold=args.loss_threshold,
            high_weight=args.loss_high_weight,
        )

    raise ValueError(f"Unknown loss: {args.loss}")


# -----------------------------
# Model factory
# -----------------------------
def build_model(args: argparse.Namespace) -> nn.Module:
    """
    Factory: create a model based on --model.

    Add new models by extending this function.
    """
    name = args.model.lower()

    if name == "unet_direct":
        # Example: uses your existing PrecipitationForecaster from unet.py
        return UNetMultiHorizon(
            input_timesteps=args.input_timesteps,
            input_channels=1,
            hidden_channels=args.hidden_channels,
            kernel_size=args.kernel_size,
            output_timesteps=args.output_timesteps,
        )

    # Example placeholders:
    # if name == "convlstm":
    #     return ConvLSTMForecaster(...)
    # if name == "transformer":
    #     return SomeTransformerModel(...)

    raise ValueError(f"Unknown model: {args.model}")


# -----------------------------
# Data
# -----------------------------
def create_dataloader(
    args: argparse.Namespace,
    split: str,
    *,
    rank: int,
    world_size: int,
) -> DataLoader:
    """
    Create a distributed DataLoader for a given split.
    """
    if split == "train":
        stride = args.train_stride
        n_steps = args.steps_per_epoch
        drop_last = True
    else:
        stride = args.val_stride
        n_steps = args.val_steps if (args.val_steps is not None and args.val_steps > 0) else None
        drop_last = False

    dataset = GSMaPZarrDataset(
        zarr_path=args.zarr_path,
        group=split,
        input_timesteps=args.input_timesteps,
        output_timesteps=args.output_timesteps,
        stride=stride,
        region=args.region,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        steps_per_epoch=n_steps,
        seed=args.seed,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == "train") or (n_steps is not None),
        drop_last=drop_last,
        seed=args.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )
    return loader


# -----------------------------
# Checkpointing
# -----------------------------
def save_checkpoint(
    *,
    rank: int,
    path: Path,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_val: float,
    train_loss: float,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    if not is_rank0(rank):
        return

    ckpt = {
        "epoch": epoch,
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val": best_val,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "args": vars(args),
    }
    torch.save(ckpt, path)
    logger.info("Checkpoint saved: %s", str(path))


def load_checkpoint(
    *,
    path: Path,
    device: torch.device,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.module.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    return start_epoch, best_val


# -----------------------------
# Trainer
# -----------------------------
@dataclass
class TrainState:
    epoch: int
    best_val: float
    patience_counter: int = 0


class Trainer:
    """
    Generic DDP trainer supporting:
      - AMP BF16 (recommended for H100)
      - DistributedSampler epochs
      - checkpoint/resume
      - TensorBoard logging on rank0
      - ReduceLROnPlateau or any scheduler
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        local_rank: int,
        rank: int,
        world_size: int,
        use_amp_bf16: bool = True,
        grad_clip_norm: Optional[float] = 0.5,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp_bf16 = use_amp_bf16
        self.grad_clip_norm = grad_clip_norm
        self.writer = writer

        model = model.to(device)
        self.model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        n = 0

        it = tqdm(self.train_loader, desc=f"Epoch {epoch} [TRAIN]") if is_rank0(self.rank) else self.train_loader

        for x, y in it:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(dtype=torch.bfloat16, enabled=self.use_amp_bf16):
                pred = self.model(x)
                loss = self.criterion(pred, y)

            loss.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.grad_clip_norm))

            self.optimizer.step()

            total_loss += float(loss.item())
            n += 1

            if is_rank0(self.rank):
                it.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(total_loss/n):.4f}")

        avg_local = (total_loss / n) if n > 0 else 0.0
        return ddp_mean(avg_local, self.device)

    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0

        it = tqdm(self.val_loader, desc=f"Epoch {epoch} [VAL]") if is_rank0(self.rank) else self.val_loader

        with torch.no_grad():
            for x, y in it:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with autocast(dtype=torch.bfloat16, enabled=self.use_amp_bf16):
                    pred = self.model(x)
                    loss = self.criterion(pred, y)

                total_loss += float(loss.item())
                n += 1

                if is_rank0(self.rank):
                    it.set_postfix(loss=f"{loss.item():.4f}")

        avg_local = (total_loss / n) if n > 0 else 0.0
        return ddp_mean(avg_local, self.device)

    def fit(
        self,
        *,
        state: TrainState,
        epochs: int,
        patience: int,
        save_dir: Path,
        save_every: int,
        args: argparse.Namespace,
    ) -> TrainState:
        for epoch in range(state.epoch, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            # Scheduler step
            if self.scheduler is not None:
                # handle ReduceLROnPlateau vs others
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]

            # Logging on rank0
            if is_rank0(self.rank) and self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("LR", lr, epoch)

            if is_rank0(self.rank):
                logger.info(
                    "Epoch %d | train=%.6f val=%.6f lr=%.3e best=%.6f",
                    epoch, train_loss, val_loss, lr, state.best_val,
                )

            # Best checkpoint
            if val_loss < state.best_val:
                state.best_val = val_loss
                state.patience_counter = 0
                save_checkpoint(
                    rank=self.rank,
                    path=save_dir / "best.pth",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_val=state.best_val,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    args=args,
                )
            else:
                state.patience_counter += 1

            # Periodic checkpoint
            if save_every > 0 and epoch % save_every == 0:
                save_checkpoint(
                    rank=self.rank,
                    path=save_dir / f"epoch_{epoch}.pth",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_val=state.best_val,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    args=args,
                )

            # Early stopping
            if state.patience_counter >= patience:
                if is_rank0(self.rank):
                    logger.info("Early stopping at epoch %d (best val=%.6f).", epoch, state.best_val)
                break

            dist.barrier()

        return state


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generic DDP Trainer (torchrun)")

    # Data
    p.add_argument("--zarr_path", type=str, required=True)
    p.add_argument("--region", type=str, default="ainpp-amazon-basin")

    # Sampling / patches
    p.add_argument("--patch_size", type=int, default=320)
    p.add_argument("--patch_stride", type=int, default=320)
    p.add_argument("--steps_per_epoch", type=int, default=2000)
    p.add_argument("--val_steps", type=int, default=1000)

    # Temporal
    p.add_argument("--input_timesteps", type=int, default=12)
    p.add_argument("--output_timesteps", type=int, default=6)
    p.add_argument("--train_stride", type=int, default=6)
    p.add_argument("--val_stride", type=int, default=18)

    # Model selection
    p.add_argument("--model", type=str, default="unet_direct")
    p.add_argument("--hidden_channels", type=int, nargs="+", default=[64, 64, 64])
    p.add_argument("--kernel_size", type=int, default=3)

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--grad_clip_norm", type=float, default=0.5)

    # Loader
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Loss
    p.add_argument("--loss", type=str, default="sharpness")
    p.add_argument("--loss_threshold", type=float, default=0.5)
    p.add_argument("--loss_high_weight", type=float, default=5.0)
    p.add_argument("--w_mse", type=float, default=1.0)
    p.add_argument("--w_l1", type=float, default=0.5)
    p.add_argument("--w_ssim", type=float, default=0.1)

    # Checkpointing / logging
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    # AMP
    p.add_argument("--amp_bf16", action="store_true", help="Enable AMP BF16 (recommended on H100).")

    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    local_rank, rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if is_rank0(rank):
        logger.info("DDP world_size=%d | batch_per_gpu=%d | effective_batch=%d", world_size, args.batch_size, args.batch_size * world_size)

    # Create output dirs on rank0 and broadcast
    if is_rank0(rank):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(args.save_dir) / f"{args.model}_{ts}"
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "logs").mkdir(exist_ok=True)

        with open(save_dir / "config.txt", "w") as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

        writer = SummaryWriter(save_dir / "logs")
        save_dir_str = str(save_dir)
    else:
        writer = None
        save_dir_str = ""

    obj_list = [save_dir_str]
    dist.broadcast_object_list(obj_list, src=0)
    save_dir = Path(obj_list[0])

    # Data
    rank0_print(rank, "Loading data...")
    train_loader = create_dataloader(args, "train", rank=rank, world_size=world_size)
    val_loader = create_dataloader(args, "validation", rank=rank, world_size=world_size)

    # Model / loss / optim / sched
    model = build_model(args)
    criterion = build_loss(args)

    scaled_lr = args.lr * np.sqrt(world_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        use_amp_bf16=args.amp_bf16,
        grad_clip_norm=args.grad_clip_norm,
        writer=writer,
    )

    # Resume
    state = TrainState(epoch=1, best_val=float("inf"))
    if args.resume:
        start_epoch, best_val = load_checkpoint(
            path=Path(args.resume),
            device=device,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
        )
        state.epoch = start_epoch
        state.best_val = best_val
        if is_rank0(rank):
            logger.info("Resumed: start_epoch=%d best_val=%.6f", state.epoch, state.best_val)

    # Fit
    state = trainer.fit(
        state=state,
        epochs=args.epochs,
        patience=args.patience,
        save_dir=save_dir,
        save_every=args.save_every,
        args=args,
    )

    if is_rank0(rank):
        if writer is not None:
            writer.close()
        logger.info("Training finished. Best val=%.6f | outputs=%s", state.best_val, str(save_dir))

    cleanup_distributed()


if __name__ == "__main__":
    main()
