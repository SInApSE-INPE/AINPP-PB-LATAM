import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from src.models.afno.blocks import AFNOBlock


class AFNO2D(nn.Module):
    def __init__(
        self,
        img_size=(880, 970),
        in_chans=6,   # 6 tempos de entrada
        out_chans=6,  # 6 tempos de saída
        embed_dim=256,
        depth=8,
        patch_size=10,
        num_blocks=8  # Parâmetro novo da AFNO
    ):
        super().__init__()
        print('img_size', img_size)
        print('patch_size', patch_size)
        print('embed_dim', embed_dim)
        print('depth', depth)
        print('num_blocks', num_blocks)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.h = self.img_size[0] // self.patch_size
        self.w = self.img_size[1] // self.patch_size
        
        # Patch Embedding (Conv2d é mais rápido que unfold)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.h, self.w, embed_dim) * 0.02)
        self.dropout = nn.Dropout(0.1)
        
        # Stack de AFNO Blocks
        self.blocks = nn.ModuleList([
            AFNOBlock(embed_dim, self.h, self.w, num_blocks=num_blocks)
            for _ in range(depth)
        ])
        
        # Head de reconstrução (Upsampling)
        self.head = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, T_in, H, W] -> O modelo espera T_in como canais
        B, T, H, W = x.shape
        
        # 1. Patch Embedding
        x = self.patch_embed(x) # [B, embed_dim, h, w]
        x = x.permute(0, 2, 3, 1) # [B, h, w, embed_dim]
        
        # 2. Positional Embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # 3. AFNO Transformer Blocks
        for blk in self.blocks:
            x = blk(x)
            
        # 4. Reconstrução
        x = x.permute(0, 3, 1, 2) # [B, embed_dim, h, w]
        x = self.head(x)          # [B, T_out, H, W]
        
        return x