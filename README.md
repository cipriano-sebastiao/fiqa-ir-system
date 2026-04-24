# FiQA-2018 Financial Information Retrieval System

A comparison of lexical (BM25) and semantic (SBERT + FAISS) retrieval over the [FiQA-2018](https://sites.google.com/view/fiqa) financial QA corpus, evaluated on the BEIR benchmark. Built as part of my MSc coursework and extended into an interactive demo.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](http://127.0.0.1:7860/)

---

## Results

Evaluated on 648 test queries from the FiQA-2018 benchmark:

| Model | k₁ | b | NDCG@10 | MAP@100 | Recall@100 | P@10 |
|---|---|---|---|---|---|---|
| BM25 Baseline | 0.9 | 0.40 | 0.2345 | 0.1874 | 0.4952 | 0.0648 |
| BM25 Tuned | 0.9 | 0.75 | 0.2401 | 0.1918 | 0.5084 | 0.0674 |
| **SBERT Dense** | — | — | **0.4443** | **0.3792** | **0.7936** | **0.1249** |

SBERT achieves **+84.6% NDCG@10** over the best BM25 configuration. The dominant factor is vocabulary mismatch. FiQA questions use conversational language while document passages use financial terminology; dense semantic encoding resolves that gap.

---

## Architecture

```
Query (natural language)
        │
        ├──── BM25 Tuned ────► bm25s tokenise ──► BM25(k₁=0.9, b=0.75) ──► Ranked docs
        │
        └──── SBERT Dense ───► multi-qa-mpnet ──► FAISS IndexFlatIP ──────► Ranked docs
                               (768-dim, fp16)      (57,638 vectors)
```

**Dataset** · FiQA-2018 via [BEIR](https://github.com/beir-cellar/beir) - 57,638 financial passages, 648 test queries  
**BM25** · [bm25s](https://github.com/xhluca/bm25s) - grid-searched k₁ ∈ {0.5, 0.9, 1.2, 1.5} × b ∈ {0.25, 0.40, 0.55, 0.75}  
**SBERT** · [`multi-qa-mpnet-base-dot-v1`](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1) - fine-tuned for asymmetric QA retrieval  
**Index** · FAISS `IndexFlatIP` with L2-normalised vectors

---

## Repository

```
fiqa-ir-system/
├── app.py                          # Gradio demo app (run locally)
├── requirements.txt                # Python dependencies
├── README.md
└── notebooks/
    └── fiqa_ir_system_final.ipynb  # Full research notebook
                                    #   Section 1 — Setup & Data Loading
                                    #   Section 2 — BM25 Baseline
                                    #   Section 3 — BM25 Hyperparameter Tuning
                                    #   Section 4 — SBERT Dense Retrieval
                                    #   Section 5 — Results Summary
                                    #   Section 6 — Interactive Gradio Demo
                                    #   Section 7 — Export for Static Demo
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<cipriano-sebastiao>/fiqa-ir-system.git
cd fiqa-ir-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
python app.py
```

The app will:
1. Download the FiQA-2018 dataset automatically (~20 MB)
2. Build the BM25 index (~10 s)
3. Encode the corpus with SBERT and build the FAISS index (~4 min CPU / ~1 min GPU)
4. Open a local Gradio interface at `http://localhost:7860`

### Running on Google Colab

Open `notebooks/fiqa_ir_system_final.ipynb` in Colab. Section 6 launches a public Gradio share link (`share=True`) valid for 72 hours.

---

## Tech Stack

| Component | Library |
|---|---|
| Dataset loading | `beir` |
| Lexical retrieval | `bm25s` |
| Dense encoding | `sentence-transformers` |
| Vector search | `faiss-cpu` / `faiss-gpu` |
| Interactive demo | `gradio` |
| Data processing | `pandas`, `numpy` |
| Deep learning | `torch` |

---

## Main Findings

- **BM25 tuning** (grid search over k₁, b) yields a modest +2.4% NDCG@10 gain. The optimal b=0.75 (stronger length normalisation) reflects the variable passage lengths in FiQA-2018.
- **Switching paradigm** from lexical to semantic retrieval yields +84.6% NDCG@10 — showing that model selection matters far more than hyperparameter tuning within a paradigm.
- **Recall@100** improves from 0.508 → 0.794 with SBERT, meaning dense retrieval finds ~56% more relevant documents in the top 100.
- The vocabulary mismatch between conversational questions and domain-specific passages is the main bottleneck for BM25 on this dataset.

---
