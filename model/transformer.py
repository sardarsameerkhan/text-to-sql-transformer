import torch
import torch.nn as nn
from model.layers import EncoderLayer, DecoderLayer


def make_pad_mask(seq, pad_idx=0):
    # Returns True for valid tokens, False for padding positions
    return (seq != pad_idx)


def make_causal_mask(seq_len, device):
    # Generates lower triangular boolean matrix (1, L, L)
    return torch.tril(torch.ones((seq_len, seq_len), device=device)).bool().unsqueeze(0)


class Encoder(nn.Module):
    def __init__(self, input_layer, num_layers=3, d_model=256, h=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.input_layer = input_layer
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, src, src_mask):
        x = self.input_layer(src)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, input_layer, num_layers=3, d_model=256, h=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.input_layer = input_layer
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, tgt, enc_out, tgt_mask, src_mask):
        x = self.input_layer(tgt)
        last_cross_attn = None
        for layer in self.layers:
            x, last_cross_attn = layer(x, enc_out, tgt_mask, src_mask)
        return x, last_cross_attn


class Seq2SeqTransformer(nn.Module):
    def __init__(self, token_emb, enc_input_layer, dec_input_layer, vocab_size, pad_id=0,
                 d_model=256, num_layers=3, h=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = token_emb

        self.encoder = Encoder(enc_input_layer, num_layers, d_model, h, d_ff, dropout)
        self.decoder = Decoder(dec_input_layer, num_layers, d_model, h, d_ff, dropout)

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        # Weight Sharing: Tie output projection weight to Token Embedding matrix
        self.output_proj.weight = self.token_emb.emb.weight

    def forward(self, src, tgt):
        device = src.device

        # Masks
        src_mask = make_pad_mask(src, self.pad_id)
        
        tgt_pad_mask = make_pad_mask(tgt, self.pad_id).unsqueeze(1)
        causal_mask = make_causal_mask(tgt.size(1), device)
        tgt_mask = tgt_pad_mask & causal_mask

        # Encoder-Decoder forward
        enc_out = self.encoder(src, src_mask)
        dec_out, cross_attn = self.decoder(tgt, enc_out, tgt_mask, src_mask)

        # Vocabulary projection
        logits = self.output_proj(dec_out)
        return logits, cross_attn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)