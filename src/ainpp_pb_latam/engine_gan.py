"""
GAN Training Engine.

This module implements the specialized training loop for Generative Adversarial Networks.
It handles the alternating optimization of the Generator and Discriminator.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ainpp_pb_latam.distributed import is_main_process
from ainpp_pb_latam.utils import save_epoch_checkpoint

logger = logging.getLogger(__name__)


def run_gan_training(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    device: torch.device,
    epochs: int,
    pixel_criterion: nn.Module,  # L1 or MSE (Content Loss)
    gan_criterion: nn.Module,  # BCE or MSE (Adversarial Loss)
    lambda_pixel: float = 100.0,  # Weight for content loss
    checkpoint_cfg: Dict = None,
    train_sampler: Optional[DistributedSampler] = None,
) -> None:
    """
    Executes the GAN training loop (Pix2Pix style).

    Args:
        generator (nn.Module): The forecasting model (U-Net, AFNO, etc.).
        discriminator (nn.Module): The PatchGAN discriminator.
        opt_g (optim.Optimizer): Optimizer for Generator.
        opt_d (optim.Optimizer): Optimizer for Discriminator.
        pixel_criterion (nn.Module): Reconstruction loss (e.g., L1Loss).
        gan_criterion (nn.Module): Adversarial loss (e.g., MSELoss for LSGAN).
        lambda_pixel (float): How much the Generator cares about pixel accuracy vs realism.
    """

    if is_main_process():
        logger.info(f"Starting GAN Training for {epochs} epochs.")

    for epoch in range(1, epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        generator.train()
        discriminator.train()

        loss_g_total = 0.0
        loss_d_total = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", disable=not is_main_process())

        for x, y in pbar:
            # x: History (B, Tin, C, H, W)
            # y: Target Future (B, Tout, C, H, W)
            x, y = x.to(device), y.to(device)

            # ------------------------------------------------------------------
            # 1. Forward Generator
            # ------------------------------------------------------------------
            # Generate fake future frames
            fake_y = generator(x)

            # Prepare inputs for Discriminator (Conditional GAN)
            # We concatenate History (x) and Future (y or fake_y) along the Channel dimension.
            # Dimensions must be permuted for Conv3d: (B, T, C, H, W) -> (B, C, T, H, W)

            # Helper to format for 3D Conv
            def format_for_d(hist, fut):
                # Concatenate along TIME dimension to create a full sequence
                # Then permute to (B, C, T, H, W)
                full_seq = torch.cat([hist, fut], dim=1)
                return full_seq.permute(0, 2, 1, 3, 4)

            # ------------------------------------------------------------------
            # 2. Update Discriminator (Maximize log(D(x, y)) + log(1 - D(x, G(x))))
            # ------------------------------------------------------------------
            opt_d.zero_grad()

            # A. Real Loss
            real_input = format_for_d(x, y)
            pred_real = discriminator(real_input)
            loss_d_real = gan_criterion(pred_real, torch.ones_like(pred_real))

            # B. Fake Loss
            # Detach fake_y because we don't want to update G here
            fake_input = format_for_d(x, fake_y.detach())
            pred_fake = discriminator(fake_input)
            loss_d_fake = gan_criterion(pred_fake, torch.zeros_like(pred_fake))

            # Total D Loss
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_d.step()

            # ------------------------------------------------------------------
            # 3. Update Generator (Maximize log(D(x, G(x))) + L1(G(x), y))
            # ------------------------------------------------------------------
            opt_g.zero_grad()

            # A. Adversarial Loss (Trick D into thinking it's real)
            fake_input_for_g = format_for_d(x, fake_y)  # No detach here!
            pred_fake_g = discriminator(fake_input_for_g)
            loss_g_adv = gan_criterion(pred_fake_g, torch.ones_like(pred_fake_g))

            # B. Content Loss (Pixel-wise accuracy)
            loss_g_pixel = pixel_criterion(fake_y, y)

            # Total G Loss
            loss_g = loss_g_adv + (lambda_pixel * loss_g_pixel)
            loss_g.backward()
            opt_g.step()

            # Logs
            loss_g_total += loss_g.item()
            loss_d_total += loss_d.item()

            if is_main_process():
                pbar.set_postfix(
                    {"G_Loss": f"{loss_g.item():.4f}", "D_Loss": f"{loss_d.item():.4f}"}
                )

        # --- End of Epoch ---
        avg_g_loss = loss_g_total / len(train_loader)
        avg_d_loss = loss_d_total / len(train_loader)

        if is_main_process():
            logger.info(f"Epoch {epoch} | G Loss: {avg_g_loss:.5f} | D Loss: {avg_d_loss:.5f}")

            if checkpoint_cfg and checkpoint_cfg.enabled and (epoch % checkpoint_cfg.interval == 0):
                # Save both models
                save_epoch_checkpoint(generator, epoch, checkpoint_cfg.dir, prefix="generator")
                save_epoch_checkpoint(
                    discriminator, epoch, checkpoint_cfg.dir, prefix="discriminator"
                )
