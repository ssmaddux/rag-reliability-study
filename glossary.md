# Glossary (Plain-English)

- **RAG (Retrieval-Augmented Generation):** The model retrieves relevant passages from a knowledge base and uses them as context to answer.
- **Embedding:** A numeric vector for text so we can compute similarity (cosine). Similar texts have vectors that point in similar directions.
- **Dense Retrieval:** Uses embeddings to find similar passages.
- **Sparse Retrieval / BM25:** Keyword-based retrieval; great for precise terms and complements dense retrieval.
- **Hybrid Retrieval:** Combine dense + BM25 results (e.g., Reciprocal Rank Fusion) for better robustness.
- **MMR (Maximal Marginal Relevance):** Selects diverse results to avoid near-duplicates dominating the top list.
- **Chunking:** Breaking long articles into smaller pieces (e.g., 512 tokens) to index and retrieve more precisely.
- **top-k:** How many passages to retrieve (e.g., the best 3 results).
- **Temperature:** How random the model is when generating text. Lower = more deterministic.
- **top-p:** Nucleus sampling; limits generation to the smallest set of tokens covering probability p.
- **Seed:** A starting number for random processes so results are repeatable.
- **Session:** Conversation state that can influence answers; “fresh session” means no prior turns.
- **Groundedness:** Whether the answer is supported by the retrieved document(s).
- **Agreement:** How similar multiple generated answers are for the same prompt (we measure via cosine similarity of embeddings).
- **MRR (Mean Reciprocal Rank):** An information-retrieval metric: how high the correct answer appears in the ranked list on average.
