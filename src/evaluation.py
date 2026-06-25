"""Framework d'evaluation et benchmarking du moteur de recherche."""

import logging
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from src.config import Config
from src.database import DatabaseManager
from src.embeddings import EmbeddingGenerator
from src.search import SemanticSearchEngine

logger = logging.getLogger(__name__)

EVAL_QUERIES = [
    {"query": "stock market crash financial crisis bank", "expected_category": "Business"},
    {"query": "football world cup championship winner", "expected_category": "Sports"},
    {"query": "artificial intelligence machine learning neural network", "expected_category": "Sci/Tech"},
    {"query": "elections president government parliament vote", "expected_category": "World"},
    {"query": "startup IPO venture capital funding billion", "expected_category": "Business"},
    {"query": "basketball NBA playoffs champion team", "expected_category": "Sports"},
    {"query": "quantum computing processor chip semiconductor", "expected_category": "Sci/Tech"},
    {"query": "war conflict military troops invasion", "expected_category": "World"},
    {"query": "merger acquisition corporate deal revenue", "expected_category": "Business"},
    {"query": "tennis grand slam wimbledon match", "expected_category": "Sports"},
    {"query": "software programming open source developer", "expected_category": "Sci/Tech"},
    {"query": "climate change global warming emission CO2", "expected_category": "World"},
    {"query": "inflation interest rate central bank monetary", "expected_category": "Business"},
    {"query": "olympic games athlete medal competition", "expected_category": "Sports"},
    {"query": "cybersecurity hacking data breach privacy", "expected_category": "Sci/Tech"},
    {"query": "united nations humanitarian crisis refugee", "expected_category": "World"},
    {"query": "oil price energy market commodity", "expected_category": "Business"},
    {"query": "racing formula grand prix circuit", "expected_category": "Sports"},
    {"query": "space exploration Mars satellite orbit", "expected_category": "Sci/Tech"},
    {"query": "diplomacy foreign policy trade agreement", "expected_category": "World"},
]


class SearchEvaluator:
    """Evaluateur de performances du moteur de recherche.

    Parameters
    ----------
    search_engine : SemanticSearchEngine
        Instance du moteur de recherche.
    """

    def __init__(self, search_engine: SemanticSearchEngine) -> None:
        self.engine = search_engine

    def evaluate_precision_at_k(self, method: str, k: int = 5) -> float:
        """Calcule la precision@k moyenne sur les requetes d'evaluation.

        Parameters
        ----------
        method : str
            'semantic' ou 'tfidf'.
        k : int
            Nombre de resultats consideres.

        Returns
        -------
        float
            Precision@k moyenne.
        """
        precisions = []
        for eq in EVAL_QUERIES:
            if method == "semantic":
                results = self.engine.search_semantic(eq["query"], top_k=k)
            else:
                results = self.engine.search_tfidf(eq["query"], top_k=k)
            correct = sum(1 for r in results if r["category"] == eq["expected_category"])
            p_at_k = correct / k
            precisions.append(p_at_k)
            logger.debug("  [%s] P@%d=%.2f — %s", method, k, p_at_k, eq["query"][:50])
        avg = np.mean(precisions)
        logger.info("P@%d (%s) = %.3f", k, method, avg)
        return float(avg)

    def evaluate_response_time(self, method: str, n_queries: int = 50) -> dict:
        """Mesure les temps de reponse.

        Parameters
        ----------
        method : str
            'semantic' ou 'tfidf'.
        n_queries : int
            Nombre de requetes de test.

        Returns
        -------
        dict
            Statistiques des temps de reponse.
        """
        with self.engine.db.conn.cursor() as cur:
            cur.execute("SELECT content FROM documents ORDER BY RANDOM() LIMIT %s", (n_queries,))
            random_texts = [r[0][:60] for r in cur.fetchall()]

        times = []
        for text in random_texts:
            start = time.perf_counter()
            if method == "semantic":
                self.engine.search_semantic(text, top_k=5)
            else:
                self.engine.search_tfidf(text, top_k=5)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        times_arr = np.array(times)
        return {
            "mean_ms": float(np.mean(times_arr)),
            "std_ms": float(np.std(times_arr)),
            "min_ms": float(np.min(times_arr)),
            "max_ms": float(np.max(times_arr)),
            "p95_ms": float(np.percentile(times_arr, 95)),
            "p99_ms": float(np.percentile(times_arr, 99)),
        }

    def run_full_benchmark(self) -> pd.DataFrame:
        """Execute le benchmark complet.

        Returns
        -------
        pd.DataFrame
            Resultats detailles par requete.
        """
        rows = []
        for eq in EVAL_QUERIES:
            row = {"query": eq["query"]}
            for method in ["semantic", "tfidf"]:
                for k in [1, 3, 5]:
                    if method == "semantic":
                        results = self.engine.search_semantic(eq["query"], top_k=k)
                    else:
                        results = self.engine.search_tfidf(eq["query"], top_k=k)
                    correct = sum(1 for r in results if r["category"] == eq["expected_category"])
                    row[f"{method}_p{k}"] = correct / k

                start = time.perf_counter()
                if method == "semantic":
                    self.engine.search_semantic(eq["query"], top_k=5)
                else:
                    self.engine.search_tfidf(eq["query"], top_k=5)
                row[f"{method}_time_ms"] = (time.perf_counter() - start) * 1000
            rows.append(row)
        return pd.DataFrame(rows)

    def generate_figures(self, output_dir: str = "report/figures/") -> None:
        """Genere les 4 graphiques de benchmark.

        Parameters
        ----------
        output_dir : str
            Repertoire de sortie des figures.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

        df = self.run_full_benchmark()

        # Figure 1 - Precision@k comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ks = [1, 3, 5]
        sem_vals = [df[f"semantic_p{k}"].mean() for k in ks]
        tfidf_vals = [df[f"tfidf_p{k}"].mean() for k in ks]
        x = np.arange(len(ks))
        ax.bar(x - 0.2, sem_vals, 0.35, label="Semantique", color="#2196F3")
        ax.bar(x + 0.2, tfidf_vals, 0.35, label="TF-IDF", color="#FF9800")
        ax.set_xticks(x)
        ax.set_xticklabels([f"P@{k}" for k in ks])
        ax.set_ylabel("Precision")
        ax.set_title("Precision@k - Recherche Semantique vs TF-IDF")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.savefig(os.path.join(output_dir, "precision_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Figure 2 - Response time boxplot
        fig, ax = plt.subplots(figsize=(10, 6))
        time_data = [df["semantic_time_ms"].values, df["tfidf_time_ms"].values]
        bp = ax.boxplot(time_data, labels=["Semantique", "TF-IDF"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#2196F3")
        bp["boxes"][1].set_facecolor("#FF9800")
        ax.set_ylabel("Temps de reponse (ms)")
        ax.set_title("Distribution des temps de reponse (20 requetes)")
        fig.savefig(os.path.join(output_dir, "response_time_boxplot.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Figure 3 - Score scatter
        fig, ax = plt.subplots(figsize=(10, 6))
        sem_scores = []
        tfidf_scores = []
        for eq in EVAL_QUERIES:
            sr = self.engine.search_semantic(eq["query"], top_k=1)
            tr = self.engine.search_tfidf(eq["query"], top_k=1)
            sem_scores.append(sr[0]["similarity_score"] if sr else 0)
            tfidf_scores.append(tr[0]["similarity_score"] if tr else 0)
        ax.scatter(sem_scores, tfidf_scores, alpha=0.7, color="#4CAF50", s=80)
        if len(sem_scores) > 1:
            z = np.polyfit(sem_scores, tfidf_scores, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(sem_scores), max(sem_scores), 100)
            corr = np.corrcoef(sem_scores, tfidf_scores)[0, 1]
            ax.plot(x_line, p(x_line), "--", color="red", label=f"r = {corr:.2f}")
            ax.legend()
        ax.set_xlabel("Score Semantique")
        ax.set_ylabel("Score TF-IDF")
        ax.set_title("Correlation des scores Semantique vs TF-IDF")
        fig.savefig(os.path.join(output_dir, "score_scatter.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Figure 4 - Category precision
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ["World", "Sports", "Business", "Sci/Tech"]
        sem_cat_p = []
        tfidf_cat_p = []
        for cat in categories:
            cat_queries = [q for q in EVAL_QUERIES if q["expected_category"] == cat]
            sem_p = []
            tfidf_p = []
            for q in cat_queries:
                sr = self.engine.search_semantic(q["query"], top_k=5)
                tr = self.engine.search_tfidf(q["query"], top_k=5)
                sem_p.append(sum(1 for r in sr if r["category"] == cat) / 5)
                tfidf_p.append(sum(1 for r in tr if r["category"] == cat) / 5)
            sem_cat_p.append(np.mean(sem_p))
            tfidf_cat_p.append(np.mean(tfidf_p))
        y = np.arange(len(categories))
        ax.barh(y - 0.2, sem_cat_p, 0.35, label="Semantique", color="#2196F3")
        ax.barh(y + 0.2, tfidf_cat_p, 0.35, label="TF-IDF", color="#FF9800")
        ax.set_yticks(y)
        ax.set_yticklabels(categories)
        ax.set_xlabel("Precision@5")
        ax.set_title("Precision@5 par categorie")
        ax.legend()
        fig.savefig(os.path.join(output_dir, "category_precision.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Figures saved to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    db = DatabaseManager()
    gen = EmbeddingGenerator()
    engine = SemanticSearchEngine(db, gen)
    evaluator = SearchEvaluator(engine)

    logger.info("Running full benchmark...")
    df = evaluator.run_full_benchmark()

    Path("report").mkdir(parents=True, exist_ok=True)
    df.to_csv("report/benchmark.csv", index=False)

    evaluator.generate_figures()

    sem_p5 = df["semantic_p5"].mean()
    tfidf_p5 = df["tfidf_p5"].mean()
    sem_time = df["semantic_time_ms"].mean()
    tfidf_time = df["tfidf_time_ms"].mean()

    table = [
        ["Semantique", f"{df['semantic_p1'].mean():.3f}", f"{df['semantic_p3'].mean():.3f}",
         f"{sem_p5:.3f}", f"{sem_time:.1f}"],
        ["TF-IDF", f"{df['tfidf_p1'].mean():.3f}", f"{df['tfidf_p3'].mean():.3f}",
         f"{tfidf_p5:.3f}", f"{tfidf_time:.1f}"],
    ]
    print("\n" + tabulate(table, headers=["Methode", "P@1", "P@3", "P@5", "Temps moy (ms)"],
                          tablefmt="rounded_grid"))

    if tfidf_p5 > 0:
        gain = ((sem_p5 - tfidf_p5) / tfidf_p5) * 100
        print(f"\nLa recherche semantique surpasse TF-IDF de {gain:.1f}% en Precision@5")

    db.close()
