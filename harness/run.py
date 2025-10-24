"""
Run experiments end-to-end.

This script:
1) Loads config (OmegaConf YAML).
2) Builds/loads indices for dense/BM25/hybrid retrieval.
3) Repeats each prompt N times (trials) to measure variance.
4) Calls the LLM backend (dummy by default) with retrieved context.
5) Logs results to CSV under results/.
6) Prints a short summary.

Teaching notes are embedded as comments.
"""

import os, json, csv, time, hashlib, random
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
from retrieval import Retriever
from llm import LLM
from eval import Evaluator
from utils import set_global_seed, ensure_dir

def main():
    cfg = OmegaConf.load("configs/base.yaml")

    # Reproducibility: set all seeds we can control.
    set_global_seed(cfg.run.seed)

    # Prepare output dir
    out_dir = Path(cfg.logging.out_dir) / f"{cfg.run.name}"
    ensure_dir(out_dir)

    # Init retrieval pipeline
    retriever = Retriever(cfg)

    # Init LLM backend
    llm = LLM(cfg)

    # Init evaluator
    evaluator = Evaluator(cfg)

    # Load prompts
    prompts = json.load(open(cfg.datasets.prompts_path, "r", encoding="utf-8"))

    rows = []
    for prompt in prompts:
        for trial in range(cfg.run.trials_per_prompt):
            # Session policy: for true "fresh" sessions, you would reset here.
            # Our dummy backend ignores sessions; real backends may carry hidden state.

            # Retrieve contexts (top_k passages)
            passages = retriever.retrieve(prompt)

            # Build a simple context string (teaching: small and explicit)
            context_text = "\n\n".join([f"[{p['ArticleNumber']}] {p['Title']}: {p['Answer']}" for p in passages])

            # Call the LLM with parameters under cfg.llm
            response = llm.generate(prompt=prompt, context=context_text)

            # Evaluate agreement/groundedness/etc.
            eval_out = evaluator.evaluate(prompt, response, passages)

            rows.append({
                "prompt": prompt,
                "trial": trial,
                "response": response,
                **eval_out
            })

    # Write CSV
    out_csv = out_dir / "results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results: {out_csv}")

if __name__ == "__main__":
    main()
