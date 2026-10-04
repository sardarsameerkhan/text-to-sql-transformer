import json
import matplotlib.pyplot as plt
import sentencepiece as spm
import torch
from embeddings import PositionalEncoding

def analyze_splits(sp):
    stats = {}
    for split in ["train", "dev", "test"]:
        with open(f"{split}_pairs.jsonl", encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
        
        src_lens = [len(sp.encode(p["src"])) + 1 for p in pairs]
        tgt_lens = [len(sp.encode(p["tgt"])) + 2 for p in pairs]
        
        dropped = sum(1 for s, t in zip(src_lens, tgt_lens) if s > 160 or t > 64)
        
        stats[split] = {
            "pairs": len(pairs),
            "mean_src": sum(src_lens) / len(src_lens),
            "max_src": max(src_lens),
            "mean_tgt": sum(tgt_lens) / len(tgt_lens),
            "max_tgt": max(tgt_lens),
            "dropped": dropped
        }
    return stats

def plot_pe_heatmap():
    pe_layer = PositionalEncoding(d_model=256, max_len=512, dropout=0.0)
    pe_matrix = pe_layer.pe[0, :100, :256].detach().cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(pe_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label="Encoding Value")
    plt.title("Positional Encoding Heat-map (First 100 Positions x 256 Dimensions)")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig("../results/pe_heatmap.png")
    print("Saved Positional Encoding Heat-map to results/pe_heatmap.png")

if __name__ == "__main__":
    sp = spm.SentencePieceProcessor(model_file="sql_sp.model")
    stats = analyze_splits(sp)
    print("\n--- Split Statistics (Table 1 Data) ---")
    print(json.dumps(stats, indent=2))
    plot_pe_heatmap()