# Reliability Checklist (v1)

## Determinism & Setup
- Temperature = 0 for eval; pin seeds end‑to‑end where supported.
- Use **fresh session** per evaluation prompt.
- Log model & embedding versions, retriever config, and index checksum.

## Retrieval & Index
- Deduplicate near‑identical articles before/after indexing.
- Prefer **hybrid retrieval (dense + BM25)** under overlapping content.
- Enable **MMR** for diversity; keep chunk size consistent (512–768) with ~20% overlap.

## Prompt & System
- Keep system prompt brief and stable.
- Canonicalize variables (IDs, dates, casing).
- Consider 2–3 run **self‑consistency** voting for high‑stakes answers.

## Monitoring & Ops
- Track agreement (semantic similarity across runs).
- Log retrieved article IDs and final cited sources.
- Rebuild embeddings periodically; version your indices.
