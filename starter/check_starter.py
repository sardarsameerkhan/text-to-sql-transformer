import sentencepiece as spm
from dataset import make_loader
from embeddings import TokenEmbedding, InputLayer
from tokenizer import PAD_ID

sp = spm.SentencePieceProcessor(model_file="sql_sp.model")
train_dl = make_loader("train_pairs.jsonl", sp, train=True)
dev_dl = make_loader("dev_pairs.jsonl", sp, train=False)

src, tgt = next(iter(train_dl))
print("src", tuple(src.shape), "tgt", tuple(tgt.shape))

d_model = 256
shared = TokenEmbedding(sp.get_piece_size(), d_model, PAD_ID)
enc_in = InputLayer(shared, d_model)
dec_in = InputLayer(shared, d_model)

x = enc_in(src)
y = dec_in(tgt[:, :-1])
print("encoder input", tuple(x.shape), "decoder input", tuple(y.shape))