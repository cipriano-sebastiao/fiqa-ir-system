"""
FiQA-2018 Financial Information Retrieval — Gradio Demo
========================================================

Demonstrates BM25 (lexical) vs SBERT (semantic) retrieval over the
57,638-document FiQA-2018 corpus.

Run locally:
    pip install -r requirements.txt
    python app.py

The app auto-downloads the FiQA-2018 dataset on first launch (~20 MB).
SBERT corpus encoding takes ~4 min on CPU or ~1 min on a T4 GPU.
All indices are cached in memory for the session.
"""

import os
import gc
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

if os.name == "nt":
    import sys, types

    if "resource" not in sys.modules:
        _res = types.ModuleType("resource")

        def _getrusage(who):
            class _R:
                ru_maxrss = 0

            return _R()

        _res.getrusage = _getrusage
        _res.RUSAGE_SELF = 0
        sys.modules["resource"] = _res

import gradio as gr
import numpy as np
import pandas as pd
import torch
import bm25s
import faiss

from sentence_transformers import SentenceTransformer
from beir import util
# Import `GenericDataLoader` from whichever path is available across
# BEIR versions. Newer releases use `data_loader`, older ones used
# `dataloader` (module name changed in some versions).
try:
    from beir.datasets.data_loader import GenericDataLoader
except Exception:
    try:
        from beir.datasets.dataloader import GenericDataLoader
    except Exception as e:
        raise ImportError(
            "Could not import GenericDataLoader from beir. "
            "Ensure `beir` is installed (e.g. `pip install beir==2.0.0`) "
            "or adjust the package version so the module path matches."
        ) from e

# ── Constants ──────────────────────────────────────────────────────────────────
# Demo model — fast on CPU (~4 min). Research notebook uses
# multi-qa-mpnet-base-dot-v1 on GPU for full benchmark results.
SBERT_MODEL = "all-MiniLM-L6-v2"
BM25_K1     = 0.9
BM25_B      = 0.75
FIQA_URL    = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
DATA_DIR    = "data"

EVAL_RESULTS = pd.DataFrame([
    {"Model": "BM25 Baseline (k₁=0.9, b=0.40)", "NDCG@10": 0.2345, "MAP@100": 0.1874, "Recall@100": 0.4952, "P@10": 0.0648},
    {"Model": "BM25 Tuned   (k₁=0.9, b=0.75)", "NDCG@10": 0.2401, "MAP@100": 0.1918, "Recall@100": 0.5084, "P@10": 0.0674},
    {"Model": "SBERT Dense  (multi-qa-mpnet)",  "NDCG@10": 0.4443, "MAP@100": 0.3792, "Recall@100": 0.7936, "P@10": 0.1249},
])

EXAMPLE_QUERIES = [
    ["How do I diversify my investment portfolio?",     "SBERT Dense", 10],
    ["What is the difference between stocks and bonds?","SBERT Dense", 10],
    ["capital gains tax rate long term investments",    "BM25 Tuned",  10],
    ["how to open a Roth IRA account",                 "BM25 Tuned",  10],
]

# ── System initialisation ─────────────────────────────────────────
print("=" * 60)
print("FiQA-2018 IR System — initialising...")
print("=" * 60)

# 1 — Dataset
print("\n[1/3] Downloading FiQA-2018 dataset...")
data_path    = util.download_and_unzip(FIQA_URL, DATA_DIR)
corpus, _, _ = GenericDataLoader(data_folder=data_path).load(split="test")

corpus_ids   = list(corpus.keys())
corpus_texts = [
    f"{corpus[d].get('title', '')} {corpus[d]['text']}".strip()
    for d in corpus_ids
]
print(f"    Corpus: {len(corpus_ids):,} documents loaded.")

# 2 — BM25
print("\n[2/3] Building BM25 index (k₁={BM25_K1}, b={BM25_B})...")
_tokens      = bm25s.tokenize(corpus_texts, stopwords=None, stemmer=None)
bm25_model   = bm25s.BM25(k1=BM25_K1, b=BM25_B)
bm25_model.index(_tokens)
print("    BM25 index ready.")

# 3 — SBERT + FAISS
print(f"\n[3/3] Loading SBERT ({SBERT_MODEL}) and encoding corpus...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"    Device: {device.upper()}")

gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

sbert_model = SentenceTransformer(SBERT_MODEL, device=device)
if device == "cuda":
    sbert_model = sbert_model.half()

corpus_embed = sbert_model.encode(
    corpus_texts,
    batch_size=256 if device == "cuda" else 64,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
)

faiss_index = faiss.IndexFlatIP(corpus_embed.shape[1])
faiss_index.add(corpus_embed)
print(f"    FAISS index: {faiss_index.ntotal:,} vectors, dim={corpus_embed.shape[1]}")

print("\nSystem ready. Launching Gradio interface...\n")

# ── Search function ────────────────────────────────────────────────────────────
def search(query: str, model: str, top_k: int) -> pd.DataFrame:
    """Return a ranked DataFrame of retrieved documents."""
    if not query.strip():
        return pd.DataFrame(columns=["Rank", "Doc ID", "Score", "Title", "Snippet"])

    top_k = int(top_k)

    if model == "BM25 Tuned":
        tokens       = bm25s.tokenize(query, stopwords=None, stemmer=None)
        idx_r, scr_r = bm25_model.retrieve(tokens, k=top_k)
        ranked = [
            (corpus_ids[idx_r[0, j]], float(scr_r[0, j]))
            for j in range(top_k)
        ]
    else:  # SBERT Dense
        q_vec        = sbert_model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )
        q_vec = q_vec.reshape(1, -1)
        scr_r, idx_r = faiss_index.search(q_vec, k=top_k)
        ranked = [
            (corpus_ids[idx_r[0, j]], float(scr_r[0, j]))
            for j in range(top_k)
        ]

    rows = []
    for rank, (doc_id, score) in enumerate(ranked, 1):
        doc     = corpus.get(doc_id, {})
        title   = (doc.get("title") or "").strip()
        snippet = doc.get("text", "")[:250].replace("\n", " ").strip()
        rows.append({
            "Rank":    rank,
            "Doc ID":  doc_id,
            "Score":   round(score, 4),
            "Title":   title or "—",
            "Snippet": snippet,
        })

    return pd.DataFrame(rows)

# ── Gradio UI ──────────────────────────────────────────────────────────────────
DESCRIPTION = """
# 🔍 FiQA-2018 Financial Information Retrieval System

Search 57,638 financial document passages using **BM25** (lexical) or **SBERT** (semantic) retrieval.  
Switch models to compare how term-matching and dense vector search differ on the same query.

| Model | NDCG@10 | MAP@100 | Recall@100 |
|-------|---------|---------|------------|
| BM25 Baseline | 0.2345 | 0.1874 | 0.4952 |
| BM25 Tuned (k₁=0.9, b=0.75) | 0.2401 | 0.1918 | 0.5084 |
| SBERT Dense (multi-qa-mpnet) | 0.4443 | 0.3792 | 0.7936 |

*Evaluated on 648 FiQA-2018 test queries · BEIR benchmark*
"""

with gr.Blocks(title="FiQA-2018 IR System", theme=gr.themes.Soft()) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        query_box = gr.Textbox(
            label="Search Query",
            placeholder="e.g. How do I diversify my investment portfolio?",
            lines=1,
            scale=4,
        )
        model_dd = gr.Dropdown(
            choices=["SBERT Dense", "BM25 Tuned"],
            value="SBERT Dense",
            label="Retrieval Model",
            scale=1,
        )
        topk_sl = gr.Slider(1, 20, value=10, step=1, label="Top-K", scale=1)

    search_btn = gr.Button("Search", variant="primary")

    results_df = gr.Dataframe(
        headers=["Rank", "Doc ID", "Score", "Title", "Snippet"],
        label="Retrieved Documents",
        interactive=False,
        wrap=True,
    )

    gr.Examples(
        examples=EXAMPLE_QUERIES,
        inputs=[query_box, model_dd, topk_sl],
    )

    search_btn.click(fn=search, inputs=[query_box, model_dd, topk_sl], outputs=results_df)
    query_box.submit(fn=search, inputs=[query_box, model_dd, topk_sl], outputs=results_df)

if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
