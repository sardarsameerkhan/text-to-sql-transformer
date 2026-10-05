import os
import sys
import json
import re
import torch
import streamlit as st
import sentencepiece as spm

# Add project root and starter directory to python path
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("starter"))

from embeddings import TokenEmbedding, InputLayer
from tokenizer import PAD_ID, BOS_ID, EOS_ID
from model.transformer import Seq2SeqTransformer, make_pad_mask, make_causal_mask


@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
    vocab_size = sp.get_piece_size()

    shared_emb = TokenEmbedding(vocab_size, 256, PAD_ID)
    enc_in = InputLayer(shared_emb, 256)
    dec_in = InputLayer(shared_emb, 256)

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

    checkpoint_path = "checkpoints/best_model.pt"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    else:
        st.error(f"Checkpoint missing at {checkpoint_path}")

    return model, sp, device


@st.cache_data
def load_dev_samples():
    dev_path = "starter/dev_pairs.jsonl"
    samples = []
    if os.path.exists(dev_path):
        with open(dev_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))
    return samples


def normalize_schema_tags(text: str) -> str:
    """Ensures spaces exist around <sep> and column tags <c0>...<c9> for SentencePiece tokenization."""
    text = re.sub(r'(<c\d+>)', r' \1 ', text)
    text = re.sub(r'(<sep>)', r' \1 ', text)
    return re.sub(r'\s+', ' ', text).strip()


def greedy_decode(model, src, max_len=100):
    model.eval()
    device = src.device
    src_mask = make_pad_mask(src, PAD_ID)
    
    with torch.no_grad():
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

            if (next_word == EOS_ID).all():
                break

    return ys


# --- Streamlit Web Interface ---
st.set_page_config(page_title="Text-to-SQL Transformer", page_icon="🔍")
st.title("🔍 Text-to-SQL Transformer Interface")
st.write("Convert Natural Language Questions into SQL Queries using a Transformer Model built from Scratch.")

model, sp, device = load_model_and_tokenizer()
dev_samples = load_dev_samples()

st.sidebar.header("Dataset Validation (dev_pairs.jsonl)")

input_mode = st.radio("Choose Input Mode:", ["Select from Dev Dataset", "Custom Query + Schema"])

if input_mode == "Select from Dev Dataset" and dev_samples:
    sample_options = [
        f"Sample {i+1}: {item.get('src', '').split('<sep>')[0].strip()}" 
        for i, item in enumerate(dev_samples[:50])
    ]
    selected_idx = st.sidebar.selectbox("Select a test sample:", range(len(sample_options)), format_func=lambda x: sample_options[x])
    
    # Direct index reference to guarantee matching item
    selected_item = dev_samples[selected_idx]
    full_src = normalize_schema_tags(selected_item.get("src", ""))
    target_sql = selected_item.get("tgt", "")

    st.text_area("Model Input Sequence (Question + Schema):", value=full_src, height=110, key=f"src_{selected_idx}")
    if target_sql:
        st.info(f"**Expected Ground Truth SQL:** `{target_sql}`")
    model_input = full_src

else:
    question = st.text_input("Natural Language Question:", "what position does the player play?")
    schema_cols = st.text_input("Table Columns (space separated):", "player number nationality position years school")

    cols = [c.strip() for c in schema_cols.replace(",", " ").split() if c.strip()]
    formatted_schema = " ".join([f"<c{i}> {col}" for i, col in enumerate(cols)])
    
    raw_input = f"{question.strip()} <sep> {formatted_schema}"
    model_input = normalize_schema_tags(raw_input)
    st.caption(f"**Formatted Input Sent to Transformer:** `{model_input}`")

if st.button("Generate SQL"):
    if model_input.strip():
        nl_text = model_input.strip().lower()
        src_tokens = [BOS_ID] + sp.encode_as_ids(nl_text) + [EOS_ID]
        src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            pred_tokens = greedy_decode(model, src_tensor)[0].tolist()

        if EOS_ID in pred_tokens:
            pred_tokens = pred_tokens[:pred_tokens.index(EOS_ID)]
        
        pred_tokens = [t for t in pred_tokens if t not in (BOS_ID, PAD_ID)]
        sql_result = sp.decode(pred_tokens)

        st.subheader("Generated SQL Query:")
        st.code(sql_result if sql_result.strip() else "SELECT ...", language="sql")
    else:
        st.warning("Please enter a valid query.")