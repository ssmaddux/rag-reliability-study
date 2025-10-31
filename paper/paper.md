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

### 4.1 Retrieval Performance and Semantic Recall Failure

We observed that the agent produced **consistent and grounded responses** across repeated trials when the relevant knowledge article (KA) closely matched the user's phrasing. For example, questions about class schedules, grade appeals, transcript requests, and password reset yielded identical responses across all trials.

However, for the query:

> **"How do I apply for student loans?"**

the agent responded with *"I don't know"* while citing **KA-000101**, despite the existence of **financial aid / FAFSA guidance** in the knowledge base.

This indicates a **semantic retrieval blind spot**:

> **The retriever failed to associate “apply for student loans” with “FAFSA” and “financial aid,” resulting in a false negative.**

Importantly, this failure occurred **consistently across all trials**, demonstrating that the limitation is **not model stochasticity**, but a **retrieval recall gap** when synonyms or paraphrasing are used.

This represents a **systematic retrieval error**, not random variance, and therefore a **core reliability risk in enterprise RAG systems**.

### 4.2 Temperature and Variance Effects

To evaluate the extent to which answer variability arises from language model stochasticity rather than retrieval behavior, we repeated the experiment at a higher decoding temperature (T=0.7). If variance were primarily due to the generative model, we would expect to see increased paraphrasing, semantic drift, or hallucinated details across repeated trials.

However, we observed that the model continued to produce **highly consistent responses** across all prompts and trials, with only minor surface-level paraphrasing (e.g., “you will need to” vs. “you should”). Importantly, the **cited Knowledge Article (KA) ID remained identical across repetitions**, indicating that the retrieval subsystem — not the language model — was the dominant factor governing answer selection.

This finding supports the interpretation introduced in Section 4.1:

> **The agent’s errors are systematic and retrieval-driven, rather than stochastic or generation-driven.**

For example, the prompt:

> **“How do I apply for student loans?”**

continued to return *“I don’t know”* with **KA-000101**, even at high temperature. This shows that **no degree of generative randomness can compensate for missing or semantically mismatched retrieval.**

Thus:

> **Retrieval stability, not LLM stochasticity, is the primary determinant of answer correctness in this system.**


### 5. Discussion
- Why the mitigations help; limits of small scale; portability to Salesforce environments.

### 6. Reliability Checklist
See `reliability_checklist.md`.

### Appendix
- Configs, prompts, dataset schema..
