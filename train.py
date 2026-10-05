import os
import sys
import math
import torch
import torch.nn as nn
from torch.optim import Adam
import sentencepiece as spm

# Include starter folder in path
sys.path.append("starter")

from dataset import make_loader
from embeddings import TokenEmbedding, InputLayer
from tokenizer import PAD_ID
from model.transformer import Seq2SeqTransformer


# --- 1. Noam Learning Rate Scheduler ---
class NoamLR:
    def __init__(self, optimizer, d_model=256, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.factor * (
            (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5))
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# --- 2. Label Smoothing Cross Entropy Loss ---
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx=0, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, x, target):
        # x: (N, C), target: (N)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist)


# --- 3. Training & Validation Loop ---
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        dec_input = tgt[:, :-1]
        targets = tgt[:, 1:]

        optimizer.zero_grad()
        logits, _ = model(src, dec_input)

        log_probs = torch.log_softmax(logits, dim=-1)
        loss = criterion(log_probs.contiguous().view(-1, log_probs.size(-1)), targets.contiguous().view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        non_pad_tokens = (targets != PAD_ID).sum().item()
        total_loss += loss.item()
        total_tokens += non_pad_tokens

    return total_loss / total_tokens


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            dec_input = tgt[:, :-1]
            targets = tgt[:, 1:]

            logits, _ = model(src, dec_input)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = criterion(log_probs.contiguous().view(-1, log_probs.size(-1)), targets.contiguous().view(-1))

            non_pad_tokens = (targets != PAD_ID).sum().item()
            total_loss += loss.item()
            total_tokens += non_pad_tokens

    return total_loss / total_tokens


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("checkpoints", exist_ok=True)

    # Data loaders
    sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
    train_dl = make_loader("starter/train_pairs.jsonl", sp, train=True, batch_size=64)
    dev_dl = make_loader("starter/dev_pairs.jsonl", sp, train=False, batch_size=64)

    vocab_size = sp.get_piece_size()
    d_model = 256

    # Instantiate shared embeddings & Transformer
    shared_emb = TokenEmbedding(vocab_size, d_model, PAD_ID)
    enc_in = InputLayer(shared_emb, d_model)
    dec_in = InputLayer(shared_emb, d_model)

    model = Seq2SeqTransformer(
        token_emb=shared_emb,
        enc_input_layer=enc_in,
        dec_input_layer=dec_in,
        vocab_size=vocab_size,
        pad_id=PAD_ID,
        d_model=256,
        num_layers=3,
        h=4,
        d_ff=1024,
        dropout=0.1
    ).to(device)

    # Optimizer, Loss, and Scheduler
    optimizer = Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, d_model=256, warmup_steps=4000)
    criterion = LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=PAD_ID, smoothing=0.1)

    epochs = 20
    best_dev_loss = float("inf")

    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_dl, criterion, optimizer, scheduler, device)
        dev_loss = evaluate(model, dev_dl, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | LR: {current_lr:.6f}")

        # Save checkpoint with lowest dev loss
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"  --> Saved new best checkpoint to checkpoints/best_model.pt")


if __name__ == "__main__":
    main()