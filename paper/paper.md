# Why Does My Agent Answer Differently? 
## Evaluating Output Variance and Retrieval Robustness in LLM‑Powered University Help Agents (Draft)

### Abstract
We study variance in answers produced by an LLM help agent for university knowledge, focusing on decoding parameters (temperature, seed, session) and retrieval under overlapping articles. On small synthetic knowledge bases that mimic Salesforce Knowledge, we find that simple settings—hybrid retrieval with near‑duplicate deduplication and MMR, paired with deterministic decoding—substantially improve stability and groundedness. We release a minimal harness and datasets to support reproducible evaluation.

### 1. Introduction
- Problem: repeated queries produce different answers; overlapping KB content confuses retrieval.
- Importance: reliability & trust in enterprise agents.
- Contributions: (1) small, reproducible harness, (2) variance & retrieval ablations, (3) practical checklist.

### 2. Related Work (brief)
- RAG stability, decoding determinism, hybrid retrieval/MMR, enterprise QA.

### 3. Methods
- Datasets: Clean vs Overlap University KB (12 articles each).
- Prompts: 12 realistic student queries.
- Retrieval: dense, BM25, hybrid (RRF), with/without MMR and near‑dup dedup.
- Decoding: temperature/top‑p/seed; session policy (fresh vs shared).
- Metrics: agreement (semantic similarity across trials), groundedness, MRR/top‑k retrieval correctness, latency.

### 4. Results (to be filled)
- Variance vs temperature and session.
- Retrieval correctness & stability under overlap; effect of MMR & dedup.
- Combined stability with best settings.

### 5. Discussion
- Why the mitigations help; limits of small scale; portability to Salesforce environments.

### 6. Reliability Checklist
See `reliability_checklist.md`.

### Appendix
- Configs, prompts, dataset schema.
