-- ============================================================
-- INDEX VECTORIELS ET CLASSIQUES
-- ============================================================

-- ============================================================
-- INDEX 1 : HNSW (Hierarchical Navigable Small World)
-- ============================================================
-- m = 16 : nombre de connexions par noeud (bon compromis memoire/recall)
-- ef_construction = 64 : taille de la liste candidats pendant construction
-- Complexite recherche : O(log n)
-- Avantage : meilleur recall que IVFFlat
-- Inconvenient : construction plus lente, memoire plus elevee
CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================
-- INDEX ALTERNATIF : IVFFlat (commente pour comparaison)
-- ============================================================
-- lists = 100 : nombre de partitions Voronoi
-- Complexite recherche : O(sqrt(n))
-- Avantage : construction plus rapide
-- Inconvenient : recall inferieur a HNSW
-- Recommande pour : datasets > 1M vecteurs avec contraintes memoire
--
-- CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat
-- ON documents USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- ============================================================
-- COMPARAISON HNSW vs IVFFlat
-- ============================================================
-- | Critere           | HNSW           | IVFFlat        |
-- |--------------------|----------------|----------------|
-- | Complexite         | O(log n)       | O(sqrt(n))     |
-- | Recall@10          | ~99%           | ~95%           |
-- | Memoire            | Elevee         | Moderee        |
-- | Construction       | Lente          | Rapide         |
-- | Cas d'usage        | < 1M vecteurs  | > 1M vecteurs  |

-- ============================================================
-- INDEX B-TREE CLASSIQUES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_documents_category
ON documents(category);

CREATE INDEX IF NOT EXISTS idx_documents_created_at
ON documents(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_search_logs_searched_at
ON search_logs(searched_at DESC);
