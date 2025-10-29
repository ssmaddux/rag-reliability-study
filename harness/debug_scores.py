"""
debug_scores.py
Prints BM25, dense, and hybrid retrieval scores for a given query.

Usage:
    python harness/debug_scores.py "How do I apply for student loans?"
"""

import sys, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import json

# --- Load your datasets ---
DATA_PATH = "datasets/overlap_knowledge.json"   # or clean_knowledge.json
with open(DATA_PATH) as f:
    articles = json.load(f)

docs = [f"{a['Title']} {a['Answer']}" for a in articles]
ids = [a['ArticleNumber'] for a in articles]

# --- Build BM25 ---
bm25 = BM25Okapi([d.lower().split() for d in docs])

# --- Build dense embeddings ---
embedder = ("sentence-transformers/all-MiniLM-L6-v2")
doc_embs = embedder.encode(docs, normalize_embeddings=True)

# --- Query embedding and tokenization ---
query = " ".join(sys.argv[1:]) or "How do I apply for student loans?"
q_tokens = query.lower().split()
q_emb = embedder.encode([query], normalize_embeddings=True)[0]

# --- Compute BM25 scores ---
bm25_scores = bm25.get_scores(q_tokens)

# --- Compute dense (cosine) scores ---
dense_scores = util.cos_sim(q_emb, doc_embs).squeeze(0).cpu().numpy()

# --- Hybrid via Reciprocal Rank Fusion (RRF) ---
def rrf_fuse(bm25_scores, dense_scores, k=60):
    bm25_ranks = np.argsort(np.argsort(-bm25_scores))
    dense_ranks = np.argsort(np.argsort(-dense_scores))
    fused = 1/(k+bm25_ranks+1) + 1/(k+dense_ranks+1)
    return fused

hybrid_scores = rrf_fuse(bm25_scores, dense_scores)

# --- Put results in a DataFrame ---
df = pd.DataFrame({
    "ArticleNumber": ids,
    "BM25": bm25_scores,
    "Dense": dense_scores,
    "Hybrid": hybrid_scores,
    "Title": [a["Title"] for a in articles],
})

# --- Sort by each metric and display ---
for col in ["BM25", "Dense", "Hybrid"]:
    print(f"\nTop 5 by {col}:\n" + "-"*60)
    top = df.sort_values(col, ascending=False).head(5)
    print(top[["ArticleNumber","Title",col]].to_string(index=False))
