from __future__ import annotations

import torch
import torch.nn as nn
import timm
from src.layers.blocks import Up
import logging

logger = logging.getLogger(__name__)


class ResNet50MultiHorizon(nn.Module):
    def __init__(self, input_timesteps=12, output_timesteps=6, pretrained=True):
        super().__init__()
        logger.info("Initializing ResNet50MultiHorizon forecaster.")
        self.encoder = timm.create_model(
            'resnet50d', 
            pretrained=pretrained, 
            in_chans=input_timesteps, 
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )
        
        # Pega dinamicamente o número de canais de cada estágio da Xception
        # Geralmente: [64, 128, 256, 728, 2048]
        enc_channels = self.encoder.feature_info.channels()
        c0, c1, c2, c3, c4 = enc_channels
        
        print(f"✅ ResNet50 Encoder Channels: {enc_channels}")

        # --- DECODER ---
        # Bottom (c4) -> Sobe para conectar com c3
        self.up1 = Up(c4, c3, skip_channels=c3)
        
        # Sobe para conectar com c2
        self.up2 = Up(c3, c2, skip_channels=c2)
        
        # Sobe para conectar com c1
        self.up3 = Up(c2, c1, skip_channels=c1)
        
        # Sobe para conectar com c0
        self.up4 = Up(c1, 64, skip_channels=c0) 
        
        # Up Final: Recupera resolução original
        # O up4 termina com resolução H/2. Precisamos de apenas UM upsample.
        # c0 na Xception tem stride 2.
        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.outc = nn.Conv2d(32, output_timesteps, kernel_size=1)

    def forward(self, x):
        # Ajuste de shape: (B, T, C, H, W) -> (B, T*C, H, W)
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x = x.view(b, t * c, h, w)
            
        original_size = x.shape[-2:]

        # --- ENCODER (timm faz tudo aqui) ---
        # Retorna lista: [x_stride_2, x_stride_4, x_stride_8, x_stride_16, x_stride_32]
        features = self.encoder(x)

        x0 = features[0] # Stride 2
        x1 = features[1] # Stride 4
        x2 = features[2] # Stride 8
        x3 = features[3] # Stride 16
        x4 = features[4] # Stride 32 (Bottom)

        # --- DECODER ---
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        
        # --- FINAL ---
        x = self.up_final(x)
        
        # Resiliência para garantir tamanho exato (320x320) e não (321x321 ou algo assim)
        if x.shape[-2:] != original_size:
            x = nn.functional.interpolate(x, size=original_size, mode='bilinear', align_corners=True)
            
        predictions = self.outc(x)
        
        # Física da chuva (não negativa)
        predictions = torch.relu(predictions)
        
        return predictions.unsqueeze(2)
