import os
import sys
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sentencepiece as spm

# Add starter directory to path for imports
sys.path.append("starter")

from dataset import make_loader
from embeddings import TokenEmbedding, InputLayer
from tokenizer import PAD_ID, BOS_ID, EOS_ID
from model.transformer import Seq2SeqTransformer, make_pad_mask, make_causal_mask


# --- 1. Decoding Algorithms ---
def greedy_decode(model, src, max_len=100):
    model.eval()
    device = src.device
    src_mask = make_pad_mask(src, PAD_ID)
    enc_out = model.encoder(src, src_mask)

    batch_size = src.size(0)
    ys = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_pad_mask = make_pad_mask(ys, PAD_ID).unsqueeze(1)
        causal_mask = make_causal_mask(ys.size(1), device)
        tgt_mask = tgt_pad_mask & causal_mask

        dec_out, _ = model.decoder(ys, enc_out, tgt_mask, src_mask)
        logits = model.output_proj(dec_out[:, -1:])
        next_word = torch.argmax(logits, dim=-1)

        ys = torch.cat([ys, next_word], dim=1)

        # Stop if EOS token is generated
        if next_word.item() == EOS_ID:
            break

    return ys


def beam_search_decode(model, src, beam_size=4, max_len=100):
    model.eval()
    device = src.device
    src_mask = make_pad_mask(src, PAD_ID)
    enc_out = model.encoder(src, src_mask)

    # Beam state: list of (sequence_ids, cumulative_log_prob)
    beams = [([BOS_ID], 0.0)]

    for _ in range(max_len - 1):
        candidates = []
        for seq, score in beams:
            if seq[-1] == EOS_ID:
                candidates.append((seq, score))
                continue

            ys = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_pad_mask = make_pad_mask(ys, PAD_ID).unsqueeze(1)
            causal_mask = make_causal_mask(ys.size(1), device)
            tgt_mask = tgt_pad_mask & causal_mask

            with torch.no_grad():
                dec_out, _ = model.decoder(ys, enc_out, tgt_mask, src_mask)
                logits = model.output_proj(dec_out[:, -1:])
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).squeeze(0)

            top_k_probs, top_k_ids = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                next_id = top_k_ids[i].item()
                next_score = score + top_k_probs[i].item()
                candidates.append((seq + [next_id], next_score))

        # Sort candidates by score descending and keep top beam_size
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

        if all(seq[-1] == EOS_ID for seq, _ in beams):
            break

    best_seq = beams[0][0]
    return torch.tensor([best_seq], dtype=torch.long, device=device)


# --- 2. Generate Predictions JSONL ---
def generate_predictions(model, sp, dev_pairs_path, output_path, decode_fn, beam_size=None):
    os.makedirs("results", exist_ok=True)
    device = next(model.parameters()).device

    with open(dev_pairs_path, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, item in enumerate(pairs):
            # Key resolution across various JSON configurations
            nl_text = item.get("question") or item.get("src") or item.get("nl")
            if not nl_text:
                continue

            src_tokens = [BOS_ID] + sp.encode_as_ids(nl_text) + [EOS_ID]
            src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)

            with torch.no_grad():
                if beam_size:
                    pred_tokens = decode_fn(model, src_tensor, beam_size=beam_size)[0].tolist()
                else:
                    pred_tokens = decode_fn(model, src_tensor)[0].tolist()

            # Clean EOS and special tokens
            if EOS_ID in pred_tokens:
                pred_tokens = pred_tokens[:pred_tokens.index(EOS_ID)]

            pred_query = sp.decode(pred_tokens)
            out_f.write(json.dumps({"query": pred_query}) + "\n")

    print(f"Generated predictions saved to: {output_path}")


# --- 3. Component Analysis ---
def run_component_analysis(pred_file, gold_file):
    if not os.path.exists(pred_file) or not os.path.exists(gold_file):
        print(f"Skipping component analysis: missing {pred_file} or {gold_file}")
        return

    with open(pred_file, "r", encoding="utf-8") as f:
        preds = [json.loads(line)["query"].upper() for line in f]

    with open(gold_file, "r", encoding="utf-8") as f:
        golds = []
        for line in f:
            item = json.loads(line)
            if isinstance(item.get("sql"), dict):
                sql_str = item["sql"].get("query", "")
            else:
                sql_str = item.get("tgt") or item.get("sql") or item.get("query", "")
            golds.append(sql_str.upper())

    sel_correct, where_correct = 0, 0
    total = min(len(preds), len(golds))

    for p, g in zip(preds[:total], golds[:total]):
        p_sel = p.split("FROM")[0] if "FROM" in p else ""
        g_sel = g.split("FROM")[0] if "FROM" in g else ""
        if p_sel == g_sel and p_sel != "":
            sel_correct += 1

        p_where = p.split("WHERE")[1] if "WHERE" in p else ""
        g_where = g.split("WHERE")[1] if "WHERE" in g else ""
        if p_where == g_where and p_where != "":
            where_correct += 1

    print("\n--- Component Analysis ---")
    print(f"SELECT Column Match: {sel_correct / total * 100:.2f}%" if total > 0 else "0.00%")
    print(f"WHERE Clause Match:  {where_correct / total * 100:.2f}%\n" if total > 0 else "0.00%\n")


# --- 4. Cross-Attention Heatmap ---
def plot_cross_attention(model, sp, src_text, tgt_text, save_path="results/cross_attention_map.png"):
    model.eval()
    device = next(model.parameters()).device

    src_ids = [BOS_ID] + sp.encode_as_ids(src_text) + [EOS_ID]
    tgt_ids = [BOS_ID] + sp.encode_as_ids(tgt_text)

    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        _, cross_attn = model(src, tgt)

    # Cross attention visualization processing
    attn_map = cross_attn.squeeze(0).mean(dim=0).cpu().numpy()

    src_tokens = [sp.id_to_piece(i) for i in src_ids]
    tgt_tokens = [sp.id_to_piece(i) for i in tgt_ids]

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_map, xticklabels=src_tokens, yticklabels=tgt_tokens, cmap="viridis")
    plt.xlabel("Source Sequence (Natural Language)")
    plt.ylabel("Target Sequence (SQL)")
    plt.title("Decoder Cross-Attention Map")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved cross-attention plot to: {save_path}")


# --- 5. Main Execution Loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
    vocab_size = sp.get_piece_size()
    d_model = 256

    # Instantiate Model
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

    # Load Model Checkpoint
    checkpoint_file = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_file}'. Check path.")

    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    print("Loaded model checkpoint successfully!")

    # 1. Greedy Search Predictions
    generate_predictions(model, sp, "starter/dev_pairs.jsonl", "results/dev_greedy.jsonl", greedy_decode)

    # 2. Beam Search Predictions (beam_size=4)
    generate_predictions(model, sp, "starter/dev_pairs.jsonl", "results/dev_beam.jsonl", beam_search_decode, beam_size=4)

    # 3. Component Analysis
    run_component_analysis("results/dev_greedy.jsonl", "starter/dev_pairs.jsonl")

    # 4. Heatmap Generation
    sample_src = "what is the player position for terrence ross"
    sample_tgt = "SELECT Position WHERE Player = Terrence Ross"
    plot_cross_attention(model, sp, sample_src, sample_tgt)


if __name__ == "__main__":
    main()