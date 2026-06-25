-- ============================================================
-- REQUETE 1 : Recherche cosinus top-5
-- ============================================================
-- L'operateur <=> calcule la distance cosinus dans pgvector.
-- similarity = 1 - distance cosinus
-- Utilise quand les vecteurs sont normalises (normalize_embeddings=True)
SELECT id, title, category,
       1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity_score
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;
-- Resultat attendu :
-- id | title                              | category | similarity_score
-- 42 | "Stock Market Hits Record High"    | Business | 0.89
-- 15 | "Wall Street Rallies on Earnings"  | Business | 0.85

-- ============================================================
-- REQUETE 2 : Recherche L2 top-5
-- ============================================================
-- L'operateur <-> calcule la distance euclidienne (L2).
-- Adapte quand les vecteurs ne sont PAS normalises.
-- Pour des vecteurs normalises, <=> et <-> donnent le meme classement.
SELECT id, title, category,
       embedding <-> '[0.1, 0.2, ...]'::vector AS l2_distance
FROM documents
ORDER BY embedding <-> '[0.1, 0.2, ...]'::vector
LIMIT 5;
-- Comparaison : <=> vs <->
-- <=> (cosinus) : invariant a la norme, mesure l'angle entre vecteurs
-- <-> (L2) : sensible a la norme, mesure la distance geometrique

-- ============================================================
-- REQUETE 3 : Recherche filtree par categorie + seuil
-- ============================================================
-- Filtre les resultats par categorie et impose un seuil de distance
-- distance < 0.3 equivaut a similarity > 0.7
SELECT id, title, content, category,
       1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM documents
WHERE category = 'Sci/Tech'
  AND (embedding <=> '[0.1, 0.2, ...]'::vector) < 0.3
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;

-- ============================================================
-- REQUETE 4 : Analyse de la qualite de l'index
-- ============================================================
-- Verifie que les index HNSW et B-tree sont bien crees
SELECT schemaname, tablename, indexname, indexdef
FROM pg_indexes
WHERE tablename = 'documents';

-- ============================================================
-- REQUETE 5 : Statistiques des temps de reponse (24h)
-- ============================================================
-- Calcule les metriques de performance des dernieres 24 heures
SELECT
    COUNT(*) AS total_searches,
    ROUND(AVG(execution_time_ms)::numeric, 2) AS avg_ms,
    ROUND(MAX(execution_time_ms)::numeric, 2) AS max_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms)::numeric, 2) AS p95_ms
FROM search_logs
WHERE searched_at > NOW() - INTERVAL '24 hours';
-- Resultat attendu :
-- total_searches | avg_ms | max_ms | p95_ms
-- 150            | 12.30  | 45.20  | 28.60

-- ============================================================
-- REQUETE 6 : Distribution des scores (histogramme SQL)
-- ============================================================
-- Repartit les scores de similarite en tranches de 0.1
-- Utile pour visualiser la qualite globale du modele
SELECT
    width_bucket(1 - (embedding <=> '[0.1, 0.2, ...]'::vector), 0, 1, 10) AS bucket,
    COUNT(*) AS count,
    ROUND(MIN(1 - (embedding <=> '[0.1, 0.2, ...]'::vector))::numeric, 3) AS min_score,
    ROUND(MAX(1 - (embedding <=> '[0.1, 0.2, ...]'::vector))::numeric, 3) AS max_score
FROM documents
GROUP BY bucket
ORDER BY bucket;

-- ============================================================
-- REQUETE 7 : Documents les plus similaires entre eux
-- ============================================================
-- Trouve les 5 plus proches voisins d'un document specifique (id=1)
-- Utile pour valider que les embeddings captent la semantique
SELECT d2.id, d2.title, d2.category,
       1 - (d1.embedding <=> d2.embedding) AS similarity
FROM documents d1
CROSS JOIN LATERAL (
    SELECT id, title, category, embedding
    FROM documents
    WHERE id != d1.id
    ORDER BY embedding <=> d1.embedding
    LIMIT 5
) d2
WHERE d1.id = 1;
-- Resultat attendu : les 5 articles les plus similaires a l'article id=1
