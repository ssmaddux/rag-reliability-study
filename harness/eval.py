"""
Evaluation utilities:
- Agreement via semantic similarity (cosine on sentence embeddings)
- Groundedness: whether the response mentions a retrieved article anchor
- (Optional) Accuracy stubs you can extend with gold answers per prompt
"""

import re, json
import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.similarity_threshold = float(cfg.eval.similarity_threshold)
        # For a fuller study, you can add gold references per prompt here.

    def _cos_sim(self, a, b):
        return float(np.dot(a, b))

    def evaluate(self, prompt: str, response: str, retrieved_passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Compute a simple groundedness signal: did the model cite an anchor like [KA-000XXX]?
        cited = re.findall(r"\[(KA-\d+)\]", response)
        cited_set = set(cited)
        retrieved_ids = set([rec.get("ArticleNumber", "") for rec in retrieved_passages])
        grounded = bool(cited_set & retrieved_ids)

        # For agreement across runs, you'll later compute similarity across responses of the same prompt.
        # Here we just embed the current response so you can aggregate later.
        vec = self.embed.encode([response], normalize_embeddings=True)[0]

        return {
            "grounded": int(grounded),
            "resp_vec": json.dumps(vec.tolist())  # serialized vector for later pairwise similarity
        }
