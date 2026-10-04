import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch_size, num_heads, seq_len, d_k)
        mask: boolean or float mask matrix
        """
        d_k = q.size(-1)
        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Replace masked positions (mask == 0 or True depending on boolean) with -1e9
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask == False, -1e9)
            else:
                scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, h=4, dropout=0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        # Projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 1. Project and split into h heads: (B, L, d_model) -> (B, h, L, d_k)
        q_proj = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        if mask is not None:
            # Broadcast mask across heads: (B, 1, 1, L_k) or (B, 1, L_q, L_k)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        # 2. Compute attention
        context, attn_weights = self.attention(q_proj, k_proj, v_proj, mask=mask)

        # 3. Concatenate and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        return self.dropout(output), attn_weights