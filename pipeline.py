from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from .config import RESULTS_SUBDIRS
from gen_class.data_loader import DataLoader, LoadedData
from gen_class.features import FeatureBuilder, FeatureSet
from gen_class.modeling import ModelResult, ModelTrainer
from gen_class.results import ResultsSaver
from gen_class.unsupervised import ClusteringResult, ProjectionResult, compute_projections, run_unsupervised


@dataclass
class PipelineOutputs:
    data: LoadedData
    features: FeatureSet
    baseline_results: Dict[str, ModelResult]
    tuned_results: Dict[str, ModelResult]
    unsupervised_results: Dict[str, ClusteringResult]
    projections: Dict[str, ProjectionResult]


class PublicationReadyClassifier:
    """High-level orchestrator that stitches together data, features, and model search."""

    def __init__(
        self,
        results_dir: str = "publication_results",
        k_genomic: int = 1000,
        k_morph: int = 500,
        n_trials: int = 30,
        model_n_jobs: Optional[int] = None,
        parallel_trials: int = 4,
        scaler_name: str = "standard",
        use_label_encoder: bool = False,
    ):
        self.results_dir = results_dir
        self.n_trials = n_trials
        self.parallel_trials = parallel_trials
        self.data_loader = DataLoader(use_label_encoder=use_label_encoder)
        self.feature_builder = FeatureBuilder(
            results_dir=results_dir,
            k_genomic=k_genomic,
            k_morph=k_morph,
            scaler_name=scaler_name,
        )
        self.feature_set: Optional[FeatureSet] = None
        self.loaded_data: Optional[LoadedData] = None
        self.model_trainer: Optional[ModelTrainer] = None
        self.model_n_jobs = model_n_jobs
        self.use_label_encoder = use_label_encoder
        self._create_directories()
        self.logger = logging.getLogger(__name__)

    def _create_directories(self) -> None:
        os.makedirs(self.results_dir, exist_ok=True)
        for sub in RESULTS_SUBDIRS:
            os.makedirs(os.path.join(self.results_dir, sub), exist_ok=True)

    def load_data(
        self,
        fasta_path: str,
        mapping_csv_path: str,
        img_prefix: str,
        label_map: Optional[dict[str, int]] = None,
        use_label_encoder: Optional[bool] = None,
    ) -> LoadedData:
        """Integrate read_data.build_sequence_dataframe directly into the loader."""
        self.logger.info("Loading data: fasta=%s mapping=%s images=%s", fasta_path, mapping_csv_path, img_prefix)
        self.loaded_data = self.data_loader.load(
            fasta_path=fasta_path,
            mapping_csv_path=mapping_csv_path,
            img_prefix=img_prefix,
            label_map=label_map,
            use_label_encoder=use_label_encoder,
        )
        return self.loaded_data

    def prepare_features(self, batch_size: int = 32) -> FeatureSet:
        if self.loaded_data is None:
            raise ValueError("Call load_data() before prepare_features().")
        self.logger.info("Preparing features (batch_size=%s)", batch_size)
        self.feature_set = self.feature_builder.build(self.loaded_data, batch_size=batch_size)
        return self.feature_set

    def train_and_tune(self) -> PipelineOutputs:
        if self.feature_set is None:
            raise ValueError("Features must be prepared before training.")

        self.logger.info("Training baselines and running Optuna tuning")
        self.model_trainer = ModelTrainer(
            feature_set=self.feature_set,
            results_dir=self.results_dir,
            model_n_jobs=self.model_n_jobs,
        )
        baseline_results = self.model_trainer.train_baselines()
        tuned_results = self.model_trainer.optimize_models(
            n_trials=self.n_trials,
            parallel_trials=self.parallel_trials,
        )
        projections = compute_projections(self.feature_set)
        self.logger.info("Computed projections: %s", list(projections.keys()))
        unsupervised_results = run_unsupervised(self.feature_set, projections)
        self.logger.info("Unsupervised results: %s", list(unsupervised_results.keys()))
        outputs = PipelineOutputs(
            data=self.loaded_data,
            features=self.feature_set,
            baseline_results=baseline_results,
            tuned_results=tuned_results,
            unsupervised_results=unsupervised_results,
            projections=projections,
        )
        ResultsSaver(self.results_dir).save_all(outputs)
        return outputs

    def run_complete_pipeline(
        self,
        fasta_path: str,
        mapping_csv_path: str,
        img_prefix: str,
        label_map: Optional[dict[str, int]] = None,
        batch_size: int = 32,
        use_label_encoder: Optional[bool] = None,
    ) -> PipelineOutputs:
        self.load_data(
            fasta_path,
            mapping_csv_path,
            img_prefix,
            label_map,
            use_label_encoder=use_label_encoder,
        )
        self.prepare_features(batch_size=batch_size)
        return self.train_and_tune()
