"""
GraphCast-style Module for Regional Precipitation Nowcasting.

This module implements an Encoder-Processor-Decoder architecture based on 
DeepMind's GraphCast, adapted for regional (flat) grids using PyTorch.

Architecture Flow:
1. Encoder: Maps Grid (Pixels) -> Latent Graph (Nodes).
2. Processor: Performs GNN Message Passing (Interaction Network).
3. Decoder: Maps Latent Graph (Nodes) -> Grid (Pixels).
"""

import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron helper."""
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphCastNet(nn.Module):
    """
    GraphCast-inspired architecture for regional forecasting.
    
    Instead of an icosahedral mesh (sphere), we use a learned latent graph
    to represent the regional atmospheric state.
    """

    def __init__(
        self,
        input_timesteps: int = 2,
        input_channels: int = 1,
        output_timesteps: int = 6,
        output_channels: int = 1,
        grid_height: int = 880,
        grid_width: int = 970,
        latent_nodes: int = 1024,  # Number of nodes in the abstract graph
        embed_dim: int = 256,
        processor_layers: int = 4,
        k_neighbors: int = 8       # Neighbors for message passing
    ):
        """
        Args:
            input_timesteps (int): Number of input time frames.
            input_channels (int): Number of channels per frame.
            output_timesteps (int): Number of output time frames.
            output_channels (int): Number of output channels per frame.
            grid_height (int): Height of the input grid.
            grid_width (int): Width of the input grid.
            latent_nodes (int): Size of the bottleneck graph (V).
            embed_dim (int): Feature dimension (D).
            processor_layers (int): Number of GNN layers.
            k_neighbors (int): Number of edges per node (sparsity).
        """
        super().__init__()
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.latent_nodes = latent_nodes
        self.embed_dim = embed_dim
        
        # 1. ENCODER (Grid -> Mesh)
        # We use a CNN to extract local features before mapping to the graph
        # This reduces the huge grid (880x970) to a manageable size.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_timesteps * input_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(64, embed_dim, kernel_size=5, stride=2, padding=2), # Downsample /4
            nn.SiLU()
        )
        
        # Learnable mapping from Downsampled Grid -> Latent Nodes
        # Reduced Grid Size: (H/4) * (W/4) -> roughly 220 * 242 = ~53k pixels
        # Mapping 53k pixels to 'latent_nodes' (e.g., 1024) requires a projector.
        # Here we use a query-based attention mechanism (Cross-Attention)
        self.encoder_projector = nn.Linear(embed_dim, embed_dim)
        self.node_queries = nn.Parameter(torch.randn(1, latent_nodes, embed_dim) * 0.02)
        
        # 2. PROCESSOR (Mesh -> Mesh)
        # A stack of Interaction Networks (GNNs)
        self.processor_layers = nn.ModuleList([
            GraphProcessorBlock(embed_dim, k_neighbors)
            for _ in range(processor_layers)
        ])
        
        # 3. DECODER (Mesh -> Grid)
        # Maps Latent Nodes back to the spatial grid
        self.decoder_projector = nn.Linear(embed_dim, embed_dim * 16) # Expand features
        
        # Upsampling back to original resolution
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, output_timesteps * output_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input (B, Tin, Cin, H, W).

        Returns:
            torch.Tensor: Output (B, Tout, Cout, H, W).
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        
        # --- 1. ENCODER ---
        # Extract features (B, embed_dim, H', W')
        features = self.feature_extractor(x)
        B, D, H_prime, W_prime = features.shape
        
        # Flatten spatial: (B, H'*W', D)
        features_flat = features.flatten(2).transpose(1, 2)
        
        # Grid2Mesh: Cross-Attention to aggregate grid info into latent nodes
        # Q: Latent Nodes, K,V: Grid Features
        # Result: (B, latent_nodes, D)
        nodes = self._grid2mesh(features_flat, self.node_queries.expand(B, -1, -1))
        
        # --- 2. PROCESSOR (GNN) ---
        # Perform message passing on the latent graph
        for layer in self.processor_layers:
            nodes = layer(nodes)
            
        # --- 3. DECODER ---
        # Mesh2Grid: Cross-Attention (or simpler broadcast) to map back
        # Here we use a simpler broadcasting approach for efficiency:
        # We query the grid positions using the processed nodes.
        
        # For simplicity in this implementation: 
        # We re-expand the nodes to the grid size using attention (Mesh -> Grid)
        grid_rec = self._mesh2grid(nodes, features_flat) # (B, H'*W', D)
        
        # Reshape back to image
        grid_rec = grid_rec.transpose(1, 2).reshape(B, D, H_prime, W_prime)
        
        # Upsample to original resolution
        out = self.upsampler(grid_rec)

        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        # Reshape (B, Tout, Cout, H, W)
        out = out.reshape(B, self.output_timesteps, -1, H, W)
        
        return out

    def _grid2mesh(self, grid_feats: torch.Tensor, node_queries: torch.Tensor) -> torch.Tensor:
        """Simple Dot-Product Attention for Grid -> Mesh."""
        # grid_feats: (B, N_grid, D)
        # node_queries: (B, N_nodes, D)
        attn = torch.bmm(node_queries, grid_feats.transpose(1, 2)) # (B, N_nodes, N_grid)
        attn = F.softmax(attn * (self.embed_dim ** -0.5), dim=-1)
        return torch.bmm(attn, grid_feats) # (B, N_nodes, D)

    def _mesh2grid(self, nodes: torch.Tensor, grid_pos_emb: torch.Tensor) -> torch.Tensor:
        """Mesh -> Grid Attention."""
        # nodes: (B, N_nodes, D)
        # grid_pos_emb: (B, N_grid, D) - using original features as query proxy
        attn = torch.bmm(grid_pos_emb, nodes.transpose(1, 2)) # (B, N_grid, N_nodes)
        attn = F.softmax(attn * (self.embed_dim ** -0.5), dim=-1)
        return torch.bmm(attn, nodes)


class GraphProcessorBlock(nn.Module):
    """
    Single GNN Layer (Message Passing).
    """
    def __init__(self, dim: int, k: int = 8):
        super().__init__()
        self.k = k
        # Edge update MLP (sender, receiver) -> edge
        self.edge_mlp = MLP(dim * 2, dim, dim)
        # Node update MLP (node, aggregated_edges) -> node
        self.node_mlp = MLP(dim * 2, dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Naive dense message passing (for demonstration).
        Ideally, use torch_geometric for sparse operations.
        """
        B, N, D = x.shape
        residual = x
        
        # 1. Create Edges (All-to-All or KNN)
        # Note: In production, pre-calculate adjacency matrix!
        # Here we simulate an aggregation step for brevity.
        
        # Simplified "Global Attention" as Message Passing
        # (Since full KNN on N=1024 is expensive to code from scratch in pure PyTorch without PyG)
        # Calculate attention/relation map
        
        # Send messages: Each node attends to others
        # Q, K, V mechanism is functionally equivalent to a fully connected GNN
        q = x
        k = x
        v = x
        
        attn = torch.bmm(q, k.transpose(1, 2)) * (D ** -0.5)
        attn = F.softmax(attn, dim=-1)
        aggregated_messages = torch.bmm(attn, v)
        
        # 2. Update Nodes
        # Concatenate original node + aggregated messages
        update = torch.cat([x, aggregated_messages], dim=-1)
        x = self.node_mlp(update)
        
        return x + residual