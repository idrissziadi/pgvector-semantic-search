-- ============================================================
-- SCHEMA : Moteur de Recherche Semantique avec pgvector
-- PostgreSQL 16 + pgvector
-- ============================================================

-- Activation de l'extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- TABLE : documents
-- Stocke les articles de presse avec leurs embeddings
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100),
    source VARCHAR(200) DEFAULT 'ag_news',
    char_count INTEGER,
    word_count INTEGER,
    embedding vector(384),  -- all-MiniLM-L6-v2 produit 384 dimensions
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uq_documents_title_source UNIQUE (title, source)
);

COMMENT ON TABLE documents IS 'Articles de presse avec leurs embeddings vectoriels';
COMMENT ON COLUMN documents.embedding IS 'Vecteur 384D genere par all-MiniLM-L6-v2';
COMMENT ON COLUMN documents.char_count IS 'Nombre de caracteres du texte nettoye';
COMMENT ON COLUMN documents.word_count IS 'Nombre de mots du texte nettoye';

-- ============================================================
-- TABLE : search_logs
-- Journal des recherches effectuees
-- ============================================================
CREATE TABLE IF NOT EXISTS search_logs (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding vector(384),
    top_k INTEGER DEFAULT 5,
    similarity_metric VARCHAR(10) DEFAULT 'cosine',
    execution_time_ms FLOAT,
    results_count INTEGER,
    searched_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE search_logs IS 'Journal des requetes de recherche';
COMMENT ON COLUMN search_logs.execution_time_ms IS 'Temps d''execution en millisecondes';

-- ============================================================
-- TABLE : evaluation_results
-- Resultats des evaluations (benchmark)
-- ============================================================
CREATE TABLE IF NOT EXISTS evaluation_results (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    method VARCHAR(20) NOT NULL,  -- 'semantic' ou 'tfidf'
    results JSONB NOT NULL,
    avg_similarity FLOAT,
    precision_at_k FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE evaluation_results IS 'Resultats des benchmarks semantic vs TF-IDF';
COMMENT ON COLUMN evaluation_results.method IS 'Methode utilisee: semantic ou tfidf';

-- ============================================================
-- VERIFICATION
-- ============================================================
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
