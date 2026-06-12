import torch
from torch import nn


class UNetWithCondition(nn.Module):
    def __init__(self, in_channels, cond_dim, out_channels, hidden_size, dropout=0.0, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.cond_proj = nn.Linear(cond_dim, hidden_size)

        # Encoder block
        self.encoder_conv1 = nn.Conv2d(in_channels, hidden_size, 3, padding=1)
        self.encoder_conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.encoder_dropout = nn.Dropout(dropout)

        # Middle block
        self.middle_conv = nn.Conv2d(hidden_size * 2, hidden_size, 3, padding=1)
        self.middle_dropout = nn.Dropout(dropout)

        # Decoder block
        self.decoder_conv1 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.decoder_conv2 = nn.Conv2d(hidden_size, out_channels, 1)
        self.decoder_dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, x_with_t: torch.Tensor, condition: torch.Tensor):
        B, _, H, W = x_with_t.shape
        cond = self.cond_proj(condition).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        cond = cond.expand(-1, -1, H, W)  # [B, C, H, W]

        # Encoder
        x = self.activation(self.encoder_conv1(x_with_t))
        x = self.encoder_dropout(x)
        enc_res = x
        x = self.activation(self.encoder_conv2(x))
        x = self.encoder_dropout(x)
        if self.use_residual:
            x = x + enc_res  # Residual connection

        # Middle
        x = torch.cat([x, cond], dim=1)
        mid_res = x
        x = self.activation(self.middle_conv(x))
        x = self.middle_dropout(x)
        if self.use_residual:
            x = x + mid_res[:, :x.size(1), :, :]  # Match channels if needed

        # Decoder
        dec_res = x
        x = self.activation(self.decoder_conv1(x))
        x = self.decoder_dropout(x)
        if self.use_residual:
            x = x + dec_res  # Residual connection

        out = self.decoder_conv2(x)
        return out
