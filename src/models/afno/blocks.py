import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# --- O BLOCO CORRETO (Real AFNO) ---
class AFNOBlock(nn.Module):
    def __init__(self, dim, h, w, num_blocks=8, sparsity_threshold=0.01):
        """
        AFNOBlock

        dim: int
            Dimensão do canal de entrada. Deve ser divisível por num_blocks. 
        h: int
            Altura da entrada.
        w: int
            Largura da entrada
        num_blocks: int
            Número de blocos. Deve ser divisível por dim. Essa é a quantidade de 
            blocos de transformação espectral, ou seja, cada bloco é um tensor 
            de pesos de transformação espectral.
        sparsity_threshold: float
            Threshold de sparsity. Essa é a proporção de pesos que serão 
            eliminados na transformação espectral. 
        """
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.num_blocks = num_blocks
        
        assert dim % num_blocks == 0, f"dim {dim} deve ser divisível por num_blocks {num_blocks}"
        self.block_size = dim // num_blocks
        self.scale = 0.02

        # --- CORREÇÃO PARA AMP/GRADSCALER ---
        # Em vez de nn.Parameter(torch.randn(..., dtype=torch.cfloat)),
        # criamos como complexo temporariamente e salvamos como VIEW REAL.
        # O GradScaler verá float32 (suportado) em vez de cfloat (não suportado).
        
        # Pesos (w1)
        w1_complex = self.scale * torch.randn(h, w // 2 + 1, num_blocks, self.block_size, self.block_size, dtype=torch.cfloat)
        self.w1 = nn.Parameter(torch.view_as_real(w1_complex))
        
        # Bias (b1)
        b1_complex = self.scale * torch.randn(h, w // 2 + 1, num_blocks, self.block_size, dtype=torch.cfloat)
        self.b1 = nn.Parameter(torch.view_as_real(b1_complex))
        
        # Norms e MLP
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1) 
        )

    def forward(self, x):
        B, H, W, C = x.shape
        residual = x
        x = self.norm1(x)
        
        # --- BLOCO ESPECTRAL ---
        # Desabilitamos autocast aqui para garantir precisão e estabilidade na FFT
        with torch.amp.autocast('cuda', enabled=False):
            # Garante float32 na entrada da FFT
            x_fp32 = x.float()
            
            # 1. FFT
            x_fft = torch.fft.rfft2(x_fp32, dim=(1, 2), norm="ortho")
            x_fft = x_fft.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
            
            # 2. Converte pesos de volta para complexo (On-the-fly)
            # O otimizador atualiza self.w1 (real), mas aqui usamos como complexo
            w1_c = torch.view_as_complex(self.w1)
            b1_c = torch.view_as_complex(self.b1)
            
            # 3. Multiplicação de Blocos
            o1 = torch.einsum('b h w n i, h w n i o -> b h w n o', x_fft, w1_c) + b1_c
            
            # 4. Soft-Thresholding
            o1 = F.relu(o1.real) + 1j * F.relu(o1.imag)
            
            # 5. Reshape e IFFT
            o1 = o1.reshape(B, H, W // 2 + 1, C)
            x = torch.fft.irfft2(o1, s=(H, W), dim=(1, 2), norm="ortho")
            
        # --- FIM DO BLOCO ESPECTRAL ---

        return x + residual + self.mlp(self.norm2(x))