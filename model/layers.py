import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_2(self.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, h=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        # 1. Self-Attention + Residual + LayerNorm
        attn_out, _ = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + attn_out)

        # 2. FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, h=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        # 1. Masked Self-Attention + Residual + LayerNorm
        attn_out, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + attn_out)

        # 2. Cross-Attention + Residual + LayerNorm
        cross_out, cross_weights = self.cross_attn(x, enc_out, enc_out, mask=src_mask)
        x = self.norm2(x + cross_out)

        # 3. FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x, cross_weights