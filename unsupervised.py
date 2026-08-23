from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
import umap

from .config import DBSCAN_EPS_VALUES, DBSCAN_MIN_SAMPLES, RANDOM_STATE
from gen_class.features import FeatureSet

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    name: str
    labels: np.ndarray
    ari: float
    silhouette: float
    extra: Dict[str, float] | None = None


@dataclass
class ProjectionResult:
    name: str
    embedding: np.ndarray
    extra: Dict[str, float] | None = None


def compute_projections(
    feature_set: FeatureSet,
    tsne_perplexity: float = 30.0,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
) -> Dict[str, ProjectionResult]:
    """Generate low-dimensional projections (PCA, t-SNE, UMAP) for visualization."""
    X = feature_set.X_train_scaled

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        random_state=RANDOM_STATE,
        init="random",
        learning_rate="auto",
    )
    X_tsne = tsne.fit_transform(X)

    umap_model = umap.UMAP(
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        random_state=RANDOM_STATE,
        metric="euclidean",
    )
    X_umap = umap_model.fit_transform(X)

    logger.info("Computed projections: PCA(2), t-SNE(2), UMAP(2)")

    return {
        "pca": ProjectionResult(
            name="PCA",
            embedding=X_pca,
            extra={"var_explained": float(np.sum(pca.explained_variance_ratio_[:2]))},
        ),
        "tsne": ProjectionResult(name="t-SNE", embedding=X_tsne, extra={"perplexity": tsne_perplexity}),
        "umap": ProjectionResult(
            name="UMAP",
            embedding=X_umap,
            extra={"n_neighbors": float(umap_neighbors), "min_dist": float(umap_min_dist)},
        ),
    }


def run_unsupervised(
    feature_set: FeatureSet,
    projections: Optional[Dict[str, ProjectionResult]] = None,
) -> Dict[str, ClusteringResult]:
    """Apply PCA + GMM and DBSCAN to the PCA projection (or provided projection)."""
    X = feature_set.X_train_scaled
    y = feature_set.y_train

    # Use provided PCA projection or compute fresh
    if projections and "pca" in projections:
        X_pca = projections["pca"].embedding
    else:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X)

    results: Dict[str, ClusteringResult] = {}

    # GMM with number of components near class count
    n_components = min(max(len(np.unique(y)), 2), 10)
    gmm = GaussianMixture(n_components=n_components, random_state=RANDOM_STATE)
    gmm_labels = gmm.fit_predict(X_pca)
    gmm_result = ClusteringResult(
        name="GMM",
        labels=gmm_labels,
        ari=adjusted_rand_score(y, gmm_labels),
        silhouette=silhouette_score(X_pca, gmm_labels) if len(set(gmm_labels)) > 1 else -1.0,
        extra={"components": float(n_components)},
    )
    results[gmm_result.name] = gmm_result
    logger.info("GMM clustering: components=%s ARI=%.3f silhouette=%.3f", n_components, gmm_result.ari, gmm_result.silhouette)

    # DBSCAN grid search over eps
    best_dbscan: Optional[ClusteringResult] = None
    for eps in DBSCAN_EPS_VALUES:
        dbscan = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES)
        labels = dbscan.fit_predict(X_pca)
        # Skip degenerate clustering
        if len(set(labels)) <= 1:
            continue
        ari = adjusted_rand_score(y, labels)
        sil = silhouette_score(X_pca, labels)
        candidate = ClusteringResult(
            name="DBSCAN",
            labels=labels,
            ari=ari,
            silhouette=sil,
            extra={"eps": float(eps)},
        )
        if best_dbscan is None or candidate.silhouette > best_dbscan.silhouette:
            best_dbscan = candidate

    if best_dbscan:
        results["DBSCAN"] = best_dbscan
        logger.info("DBSCAN best eps=%.3f ARI=%.3f silhouette=%.3f", best_dbscan.extra["eps"], best_dbscan.ari, best_dbscan.silhouette)

    return results
