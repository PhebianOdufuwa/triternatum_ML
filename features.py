from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, Normalizer
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from .config import RANDOM_STATE, TEST_SIZE
from gen_class.data_loader import LoadedData

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    feature_names: list[str]
    X_train: np.ndarray
    X_test: np.ndarray
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    selected_features: list[str]


class FeatureBuilder:
    """Combine genomic and morphological features and perform lightweight selection."""

    def __init__(
        self,
        results_dir: str,
        k_genomic: int = 1000,
        k_morph: int = 500,
        random_state: int = RANDOM_STATE,
        test_size: float = TEST_SIZE,
        scaler_name: str = "standard",
    ):
        self.results_dir = results_dir
        self.k_genomic = k_genomic
        self.k_morph = k_morph
        self.random_state = random_state
        self.test_size = test_size
        self.scaler_name = scaler_name
        self.resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    def _build_scaler(self):
        name = self.scaler_name.lower()
        if name in {"none", "identity", "no"}:
            return None
        if name == "standard":
            return StandardScaler()
        if name == "robust":
            return RobustScaler()
        if name == "minmax":
            return MinMaxScaler()
        if name == "power":
            return PowerTransformer(method="yeo-johnson")
        if name == "quantile":
            return QuantileTransformer(output_distribution="normal", random_state=self.random_state)
        if name == "normalizer":
            return Normalizer()
        raise ValueError(
            "Unknown scaler_name. Use one of: none, standard, robust, minmax, power, quantile, normalizer."
        )

    def _extract_resnet(self, images: tf.Tensor, batch_size: int = 32) -> np.ndarray:
        """Turn image tensors into ResNet embeddings."""
        logger.info("Extracting ResNet embeddings for %s images", images.shape[0])
        processed = resnet_preprocess(images * 255.0)
        return self.resnet.predict(processed, batch_size=batch_size, verbose=1)

    def build(
        self,
        data: LoadedData,
        batch_size: int = 32,
    ) -> FeatureSet:
        if data is None:
            raise ValueError("Data must be loaded before building features.")

        # Deep features from images
        morph_features = self._extract_resnet(data.images, batch_size=batch_size)

        # DataFrames for easy column tracking
        genomic_df = pd.DataFrame(
            data.genomic_matrix, columns=[f"genome_{i}" for i in range(data.genomic_matrix.shape[1])]
        )
        morph_df = pd.DataFrame(morph_features, columns=[f"resnet_{i}" for i in range(morph_features.shape[1])])

        combined = pd.concat([genomic_df, morph_df], axis=1)
        logger.info("Combined feature matrix shape: %s", combined.shape)

        # Feature selection applied separately to each modality using training data only.
        genome_cols = [c for c in combined.columns if c.startswith("genome_")]
        morph_cols = [c for c in combined.columns if c.startswith("resnet_")]

        X = combined.to_numpy()
        y = data.labels

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        genome_idx = [combined.columns.get_loc(c) for c in genome_cols]
        morph_idx = [combined.columns.get_loc(c) for c in morph_cols]

        genome_selector = SelectKBest(f_classif, k=min(self.k_genomic, len(genome_cols)))
        morph_selector = SelectKBest(f_classif, k=min(self.k_morph, len(morph_cols)))

        genome_selected_train = genome_selector.fit_transform(X_train[:, genome_idx], y_train)
        morph_selected_train = morph_selector.fit_transform(X_train[:, morph_idx], y_train)
        genome_selected_test = genome_selector.transform(X_test[:, genome_idx])
        morph_selected_test = morph_selector.transform(X_test[:, morph_idx])

        selected_cols = list(genome_cols[i] for i, mask in enumerate(genome_selector.get_support()) if mask)
        selected_cols += list(morph_cols[i] for i, mask in enumerate(morph_selector.get_support()) if mask)

        X_train_selected = np.hstack([genome_selected_train, morph_selected_train])
        X_test_selected = np.hstack([genome_selected_test, morph_selected_test])
        logger.info(
            "Selected features - genomic: %s, morph: %s, total: %s",
            genome_selected_train.shape[1],
            morph_selected_train.shape[1],
            X_train_selected.shape[1],
        )

        scaler = self._build_scaler()
        if scaler is None:
            X_train_scaled = X_train_selected
            X_test_scaled = X_test_selected
        else:
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)

        return FeatureSet(
            feature_names=selected_cols,
            X_train=X_train_selected,
            X_test=X_test_selected,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            selected_features=selected_cols,
        )
