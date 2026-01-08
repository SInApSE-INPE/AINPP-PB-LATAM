
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.convlstm.backbone import ConvLSTM2D


logger = logging.getLogger(__name__)


class ConvLSTMMultiHorizon(nn.Module):
    """
    Modelo completo para previsão de precipitação multi-lead time.
    Arquitetura Encoder-Decoder com ConvLSTM. 
    
    INPUT:  Dados NRT (Near Real-Time - brutos do satélite)
    OUTPUT: Dados MVK (Moving Vector with Kalman - calibrados)
    """
    def __init__(
        self,
        input_channels: int =1,
        hidden_channels: List[int] =[64, 64, 64],
        kernel_size: int =7,
        output_timesteps: int =6,
    ) -> None:
        super(ConvLSTMMultiHorizon, self).__init__()
        logger.info("Initializing ConvLSTMMultiHorizon forecaster.")
        
        self.input_channels = input_channels
        self.output_timesteps = output_timesteps
        
        # Encoder:  processa sequência de entrada (NRT)
        self.encoder = ConvLSTM2D(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            # kernel_size=7,
        )
        
        # Decoder:  gera predições futuras (MVK calibrado)
        self.decoder = ConvLSTM2D(
            input_channels=hidden_channels[-1],  # Recebe output do encoder
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        
        # Camada de saída: converte features para 1 canal (precipitação MVK)
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  # 1 canal de saída (MVK)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            x: (B, T_in, 1, H, W) - sequência de entrada (NRT)
        Returns:
            predictions: (B, T_out, 1, H, W) - previsões futuras (MVK)
        """
        batch_size = x.size(0)
        
        # FASE 1: ENCODING (processa dados NRT)
        encoder_outputs, encoder_states = self.encoder(x)
        
        # FASE 2: DECODING (gera predições MVK calibradas)
        # Usa o último output do encoder como primeiro input do decoder
        decoder_input = encoder_outputs[-1][: , -1:, : , : , :]  # (B, 1, C, H, W)
        
        # Inicializa decoder com estados finais do encoder
        decoder_hidden = encoder_states
        
        predictions = []
        
        # Gera previsões autoregressivamente
        for t in range(self.output_timesteps):
            # Passa input pelo decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Pega output da última camada do decoder
            current_output = decoder_output[-1][: , -1, :, :, :]  # (B, C, H, W)
            
            # Converte para previsão de precipitação MVK
            prediction = self.output_conv(current_output)  # (B, 1, H, W)
            predictions.append(prediction)
            
            # Usa previsão como próximo input (autoregressivo)
            decoder_input = current_output. unsqueeze(1)  # (B, 1, C, H, W)
        
        # Empilha todas as previsões
        predictions = torch. stack(predictions, dim=1)  # (B, T_out, 1, H, W)
        
        return predictions
