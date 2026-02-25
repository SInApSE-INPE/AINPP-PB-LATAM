from __future__ import annotations

import logging
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ainpp.models.convlstm.blocks import ConvLSTMCell


logger = logging.getLogger(__name__)


class ConvLSTM2D(nn.Module):
    """
    ConvLSTM multi-camada para previsão de precipitação.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTM2D, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self. kernel_size = kernel_size
        self.num_layers = len(self.hidden_channels)

        # Criar células para cada camada
        cell_list = []
        for i in range(self.num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_channels=cur_input_channels,
                    hidden_channels=hidden_channels[i],
                    kernel_size=kernel_size
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: (B, T, C, H, W) - sequência de entrada
            hidden_state: lista de tuplas [(h, c)] para cada camada
        Returns: 
            layer_output_list: lista com outputs de cada camada
            last_state_list: lista com últimos estados de cada camada
        """
        batch_size, seq_len, _, height, width = x.size()
        
        # Inicializar hidden states se não fornecidos
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Processar cada timestep
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, : ], (h, c))
                output_inner.append(h)

            # Empilhar outputs temporais
            layer_output = torch.stack(output_inner, dim=1)  # (B, T, C, H, W)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """Inicializa hidden states para todas as camadas."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self. cell_list[i].init_hidden(batch_size, image_size))
        return init_states