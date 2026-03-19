import torch
import torch.nn as nn
from typing import List, Tuple


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell - processes one timestep maintaining spatial structure.
    """

    def __init__(self, input_channels: int, hidden_channels: List[int], kernel_size: int) -> None:
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Combined convolution for all gates (i, f, o, g)
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,  # 4 gates
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self, x: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C_in, H, W) - entrada atual
            hidden_state: tuple (h, c) where h and c are (B, C_hidden, H, W)
        Returns:
            h_next, c_next: next hidden and cell states
        """
        h_cur, c_cur = hidden_state

        # Concatenates input with previous hidden state
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)

        # Split nas 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)

        # Apply activation functions
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)  # Candidate cell state

        # Atualizar cell state e hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self, batch_size: int, image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inicializa estados hidden e cell com zeros."""
        height, width = image_size
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
        )
