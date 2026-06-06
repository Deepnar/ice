
CREATE INDEX IF NOT EXISTS idx_episodic_embedding
ON episodic_memory USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_procedural_embedding
ON procedural_memory USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding
ON rag_chunks USING hnsw (embedding vector_cosine_ops);