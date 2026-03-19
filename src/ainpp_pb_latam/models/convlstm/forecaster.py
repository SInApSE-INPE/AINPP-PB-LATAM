from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ainpp_pb_latam.models.convlstm.backbone import ConvLSTM2D

logger = logging.getLogger(__name__)


class ConvLSTMMultiHorizon(nn.Module):
    """
    Complete model for multi-lead time precipitation forecasting.
    Arquitetura Encoder-Decoder com ConvLSTM.

    INPUT:  NRT Data (Near Real-Time - raw satellite data)
    OUTPUT: Dados MVK (Moving Vector with Kalman - calibrados)
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: List[int] = [64, 64, 64],
        kernel_size: int = 7,
        output_timesteps: int = 6,
    ) -> None:
        super(ConvLSTMMultiHorizon, self).__init__()
        logger.info("Initializing ConvLSTMMultiHorizon forecaster.")

        self.input_channels = input_channels
        self.output_timesteps = output_timesteps

        # Encoder:  processa input sequence (NRT)
        self.encoder = ConvLSTM2D(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            # kernel_size=7,
        )

        # Decoder:  generates future predictions (calibrated MVK)
        self.decoder = ConvLSTM2D(
            input_channels=hidden_channels[-1],  # Recebe output do encoder
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        # Output layer: converts features to 1 channel (MVK precipitation)
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),  # 1 output channel (MVK)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_in, 1, H, W) - input sequence (NRT)
        Returns:
            predictions: (B, T_out, 1, H, W) - future predictions (MVK)
        """
        batch_size = x.size(0)

        # PHASE 1: ENCODING (processes NRT data)
        encoder_outputs, encoder_states = self.encoder(x)

        # PHASE 2: DECODING (generates calibrated MVK predictions)
        # Uses the last output from encoder as first input for decoder
        decoder_input = encoder_outputs[-1][:, -1:, :, :, :]  # (B, 1, C, H, W)

        # Initializes decoder with final states of encoder
        decoder_hidden = encoder_states

        predictions = []

        # Generates predictions autoregressively
        for t in range(self.output_timesteps):
            # Passes input through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # Gets output of the last decoder layer
            current_output = decoder_output[-1][:, -1, :, :, :]  # (B, C, H, W)

            # Converts to MVK precipitation prediction
            prediction = self.output_conv(current_output)  # (B, 1, H, W)
            predictions.append(prediction)

            # Uses prediction as next input (autoregressive)
            decoder_input = current_output.unsqueeze(1)  # (B, 1, C, H, W)

        # Stacks all predictions
        predictions = torch.stack(predictions, dim=1)  # (B, T_out, 1, H, W)

        return predictions
