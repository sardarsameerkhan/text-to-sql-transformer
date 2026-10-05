import sys
import torch
import sentencepiece as spm

# Add starter folder to python path
sys.path.append("starter")

from dataset import make_loader
from embeddings import TokenEmbedding, InputLayer
from tokenizer import PAD_ID
from model.transformer import Seq2SeqTransformer, count_parameters

# Load tokenizer and sample data
sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
train_dl = make_loader("starter/train_pairs.jsonl", sp, train=True, batch_size=64)
src, tgt = next(iter(train_dl))

vocab_size = sp.get_piece_size()
d_model = 256

# Initialize model components
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
)

dec_input = tgt[:, :-1]
logits, _ = model(src, dec_input)

print("--- Task 2 Verification ---")
print(f"Logits output shape: {tuple(logits.shape)}")
print(f"Weight sharing verified (is identity): {model.output_proj.weight is shared_emb.emb.weight}")
print(f"Total Trainable Parameters: {count_parameters(model):,}")