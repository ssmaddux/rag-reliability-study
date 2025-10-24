"""
Retrieval pipeline illustrating:
- Dense retrieval (embeddings + FAISS)
- Sparse retrieval (BM25)
- Hybrid retrieval (reciprocal rank fusion)
- Optional MMR (diversity) and near-duplicate deduplication

All steps are deliberately small and commented to teach concepts.
"""

import os, json, math, hashlib
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from rank_bm25 import BM25Okapi

class Retriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.kb = json.load(open(cfg.datasets.knowledge_path, "r", encoding="utf-8"))
        self.top_k = cfg.retrieval.top_k
        self.type = cfg.retrieval.type

        # Prepare fields to index (simple: Title + Answer concatenation)
        self.docs = [f"{rec['Title']}\n{rec['Answer']}" for rec in self.kb]

        # Simple near-duplicate dedup at article level (cosine threshold on embeddings)
        # We'll implement as a post-retrieval collapse for simplicity.

        # Dense model (small, CPU-friendly)
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.doc_vecs = self.embed_model.encode(self.docs, normalize_embeddings=True)

        # BM25 sparse index
        tokenized = [doc.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized)
        self.tokenized = tokenized

    def _dense_search(self, query_vec, k):
        sims = np.dot(self.doc_vecs, query_vec)
        idxs = np.argsort(-sims)[:k*4]  # get a few extra for MMR/dedup
        return [(int(i), float(sims[i])) for i in idxs]

    def _bm25_search(self, query, k):
        scores = self.bm25.get_scores(query.lower().split())
        idxs = np.argsort(-scores)[:k*4]
        return [(int(i), float(scores[i])) for i in idxs]

    def _mmr(self, cand: List[tuple], query_vec, lambda_=0.7, k=3):
        selected = []
        cand_idxs = [i for i,_ in cand]
        cand_vecs = self.doc_vecs[cand_idxs]
        sim_query = np.array([np.dot(v, query_vec) for v in cand_vecs])
        selected_set = set()

        for _ in range(min(k, len(cand_idxs))):
            mmr_scores = []
            for j, idx in enumerate(cand_idxs):
                if idx in selected_set:
                    mmr_scores.append(-1e9)
                    continue
                # diversity term: distance from selected
                if not selected:
                    diversity = 0.0
                else:
                    vj = cand_vecs[j]
                    diversity = max(np.dot(vj, self.doc_vecs[s]) for s in selected)
                mmr_score = lambda_ * sim_query[j] - (1 - lambda_) * diversity
                mmr_scores.append(mmr_score)
            pick = cand_idxs[int(np.argmax(mmr_scores))]
            selected.append(pick)
            selected_set.add(pick)
        return selected

    def _dedup(self, idxs, threshold=0.92):
        # Collapse near-duplicates by semantic similarity among selected docs
        unique = []
        for idx in idxs:
            vec = self.doc_vecs[idx]
            if all(np.dot(vec, self.doc_vecs[u]) < threshold for u in unique):
                unique.append(idx)
        return unique

    def _rrf(self, dense_scores: Dict[int,float], bm25_scores: Dict[int,float], k):
        # Reciprocal Rank Fusion: rank fusion of two lists
        # Convert scores to ranks
        def ranks(score_dict):
            # higher score -> better rank
            sorted_items = sorted(score_dict.items(), key=lambda x: -x[1])
            return {doc: (rank+1) for rank, (doc, _) in enumerate(sorted_items)}

        r_d = ranks(dense_scores)
        r_s = ranks(bm25_scores)
        # fuse: 1/(k + rank)
        fused = {}
        for doc in set(list(r_d.keys()) + list(r_s.keys())):
            fused[doc] = 1.0/(60 + r_d.get(doc, 1e9)) + 1.0/(60 + r_s.get(doc, 1e9))
        # return top candidates
        return [doc for doc, _ in sorted(fused.items(), key=lambda x: -x[1])]

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        # Encode query for dense
        q_vec = self.embed_model.encode([query], normalize_embeddings=True)[0]

        # Get candidates
        dense_raw = self._dense_search(q_vec, self.top_k)
        bm25_raw  = self._bm25_search(query, self.top_k)

        # Map to dicts for fusion
        dense_scores = {i:s for i,s in dense_raw}
        bm25_scores  = {i:s for i,s in bm25_raw}

        if self.type == "dense":
            cand = [i for i,_ in dense_raw]
        elif self.type == "bm25":
            cand = [i for i,_ in bm25_raw]
        else:  # hybrid
            cand = self._rrf(dense_scores, bm25_scores, self.top_k)

        # Apply MMR (diversity) on candidate pool using dense similarities
        if self.cfg.retrieval.mmr:
            # Build candidate list as tuples with dense score for stability
            c = [(i, dense_scores.get(i, 0.0)) for i in cand]
            idxs = self._mmr(c, q_vec, k=self.top_k)
        else:
            idxs = cand[: self.top_k*2]

        # Deduplicate near-duplicates
        if self.cfg.retrieval.dedup:
            idxs = self._dedup(idxs)

        # Truncate to top_k
        idxs = idxs[: self.top_k]

        # Return structured passages
        out = []
        for i in idxs:
            rec = self.kb[i]
            out.append(rec)
        return out
