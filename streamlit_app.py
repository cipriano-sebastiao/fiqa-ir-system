"""
FiQA-2018 Financial Information Retrieval — Streamlit Demo
===========================================================

An employer-facing portfolio demo comparing BM25 (lexical) and
SBERT (semantic) retrieval over 57,638 FiQA-2018 financial passages.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploy free on Streamlit Cloud:
    https://share.streamlit.io  (connect your GitHub repo, set main file to streamlit_app.py)
"""

import os
import gc
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import bm25s
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from beir import util
from beir.datasets.dataloader import GenericDataLoader

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="FiQA-2018 · IR System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
SBERT_MODEL = "multi-qa-mpnet-base-dot-v1"
BM25_K1     = 0.9
BM25_B      = 0.75
FIQA_URL    = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
DATA_DIR    = "data"

EVAL_DF = pd.DataFrame([
    {"Model": "BM25 Baseline", "k1": 0.9, "b": 0.40, "NDCG@10": 0.2345, "MAP@100": 0.1874, "Recall@100": 0.4952, "P@10": 0.0648},
    {"Model": "BM25 Tuned",    "k1": 0.9, "b": 0.75, "NDCG@10": 0.2401, "MAP@100": 0.1918, "Recall@100": 0.5084, "P@10": 0.0674},
    {"Model": "SBERT Dense",   "k1": "—", "b": "—",  "NDCG@10": 0.4443, "MAP@100": 0.3792, "Recall@100": 0.7936, "P@10": 0.1249},
])

EXAMPLES = [
    "How do I diversify my investment portfolio?",
    "What is the difference between stocks and bonds?",
    "capital gains tax rate long term investments",
    "how to open a Roth IRA account",
    "best strategies for retirement savings",
    "should I pay off debt or invest first",
]

# ── System initialisation ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    """Download FiQA-2018, build BM25 index, encode corpus, build FAISS index.
    Cached for the entire Streamlit session — runs only once."""

    # 1 — Dataset
    data_path    = util.download_and_unzip(FIQA_URL, DATA_DIR)
    corpus, _, _ = GenericDataLoader(data_folder=data_path).load(split="test")
    corpus_ids   = list(corpus.keys())
    corpus_texts = [
        f"{corpus[d].get('title', '')} {corpus[d]['text']}".strip()
        for d in corpus_ids
    ]

    # 2 — BM25 (tuned parameters from grid search in research notebook)
    tokens     = bm25s.tokenize(corpus_texts, stopwords=None, stemmer=None)
    bm25_model = bm25s.BM25(k1=BM25_K1, b=BM25_B)
    bm25_model.index(tokens)

    # 3 — SBERT + FAISS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    sbert = SentenceTransformer(SBERT_MODEL, device=device)
    if device == "cuda":
        sbert = sbert.half()

    embeddings = sbert.encode(
        corpus_texts,
        batch_size=256 if device == "cuda" else 64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return corpus, corpus_ids, bm25_model, sbert, index


# ── Search ─────────────────────────────────────────────────────────────────────
def run_search(query, model_choice, top_k, corpus, corpus_ids, bm25_model, sbert, faiss_index):
    if not query.strip():
        return pd.DataFrame()

    if model_choice == "BM25 Tuned":
        tokens       = bm25s.tokenize(query, stopwords=None, stemmer=None)
        idx_r, scr_r = bm25_model.retrieve(tokens, k=top_k)
        ranked = [(corpus_ids[idx_r[0, j]], float(scr_r[0, j])) for j in range(top_k)]
    else:
        q_vec        = sbert.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        scr_r, idx_r = faiss_index.search(q_vec, k=top_k)
        ranked = [(corpus_ids[idx_r[0, j]], float(scr_r[0, j])) for j in range(top_k)]

    rows = []
    for rank, (doc_id, score) in enumerate(ranked, 1):
        doc     = corpus.get(doc_id, {})
        title   = (doc.get("title") or "").strip()
        snippet = doc.get("text", "")[:280].replace("\n", " ").strip()
        rows.append({
            "Rank":    rank,
            "Score":   round(score, 4),
            "Title":   title or "—",
            "Snippet": snippet,
            "Doc ID":  doc_id,
        })
    return pd.DataFrame(rows)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## FiQA-2018 IR System")
    st.caption("BM25 - SBERT Dense")
    st.divider()

    model_choice = st.radio(
        "**Retrieval Model**",
        ["SBERT Dense", "BM25 Tuned"],
        index=0,
        help=(
            "**SBERT Dense** — semantic vector search via FAISS\n\n"
            "**BM25 Tuned** — lexical term-frequency matching"
        ),
    )

    top_k = st.slider("**Top-K results**", min_value=1, max_value=20, value=10)

    st.divider()
    st.markdown("### Benchmark Results")
    st.caption("648 FiQA-2018 test queries · BEIR")

    col1, col2 = st.columns(2)
    col1.metric("BM25 NDCG@10",  "0.2401")
    col2.metric("SBERT NDCG@10", "0.4443", delta="+84.6%")

    with st.expander("Full metric table"):
        st.dataframe(EVAL_DF.set_index("Model"), use_container_width=True)

    st.divider()
    st.markdown(
        "**Dataset:** [FiQA-2018](https://sites.google.com/view/fiqa) via BEIR  \n"
        "**Encoder:** `multi-qa-mpnet-base-dot-v1`  \n"
        "**Index:** FAISS `IndexFlatIP`  \n"
        "**Corpus:** 57,638 passages"
    )


# ── Main area ──────────────────────────────────────────────────────────────────
st.title("FiQA-2018 Financial Information Retrieval")
st.markdown(
    "Search **57,638 financial document passages** using lexical or semantic retrieval. "
    "Use the sidebar to switch models and compare how BM25 and SBERT rank the same query differently."
)

# Loading
with st.spinner("Initialising search engine — ~4 min on CPU, ~1 min on GPU (one-time cost per session)…"):
    corpus, corpus_ids, bm25_model, sbert_model, faiss_idx = load_system()

st.success(f"Ready — {len(corpus_ids):,} documents indexed with both BM25 and SBERT")
st.divider()

# Search input — support both typing and example buttons
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

query = st.text_input(
    "Enter a financial query",
    value=st.session_state.query_input,
    placeholder="e.g. How do I diversify my investment portfolio?",
    label_visibility="collapsed",
    key="query_box",
)

# Example query buttons
with st.expander("Try an example query"):
    btn_cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if btn_cols[i % 3].button(ex, use_container_width=True, key=f"ex_{i}"):
            st.session_state.query_input = ex
            st.rerun()

# Results
if query.strip():
    with st.spinner(f"Searching with **{model_choice}**…"):
        df = run_search(query, model_choice, top_k,
                        corpus, corpus_ids, bm25_model, sbert_model, faiss_idx)

    st.markdown(f"### Results — *{model_choice}*")
    st.caption(f"Query: `{query}` · Top {top_k} of {len(corpus_ids):,} documents")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank":    st.column_config.NumberColumn("Rank",    width="small"),
            "Score":   st.column_config.NumberColumn("Score",   format="%.4f", width="small"),
            "Title":   st.column_config.TextColumn("Title",     width="medium"),
            "Snippet": st.column_config.TextColumn("Snippet",   width="large"),
            "Doc ID":  st.column_config.TextColumn("Doc ID",    width="small"),
        },
    )

    # Side-by-side comparison
    with st.expander("Compare both models side by side on this query"):
        other = "BM25 Tuned" if model_choice == "SBERT Dense" else "SBERT Dense"
        with st.spinner(f"Running {other}…"):
            df2 = run_search(query, other, top_k,
                             corpus, corpus_ids, bm25_model, sbert_model, faiss_idx)
        c1, c2 = st.columns(2)
        c1.markdown(f"**{model_choice}**")
        c1.dataframe(df[["Rank", "Score", "Title", "Snippet"]],
                     use_container_width=True, hide_index=True)
        c2.markdown(f"**{other}**")
        c2.dataframe(df2[["Rank", "Score", "Title", "Snippet"]],
                     use_container_width=True, hide_index=True)
else:
    st.info("👆 Type a query or click an example above to begin.")
