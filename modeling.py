from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import optuna
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from .config import RANDOM_STATE
from gen_class.evaluation import compute_metrics
from gen_class.features import FeatureSet

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    model: object
    metrics: Dict[str, float]
    best_params: Optional[Dict] = None
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    y_proba: Optional[np.ndarray] = None


class ModelTrainer:
    """Train baseline models and run Optuna-based hyperparameter search."""

    def __init__(
        self,
        feature_set: FeatureSet,
        results_dir: str,
        model_n_jobs: Optional[int] = None,
        cv_folds: int = 5,
    ):
        self.feature_set = feature_set
        self.results_dir = results_dir
        self.model_n_jobs = model_n_jobs or max((os.cpu_count() or 2) - 1, 1)
        self.cv_folds = cv_folds
        self.models: Dict[str, ModelResult] = {}
        self.label_encoder: Optional[LabelEncoder] = None

    def _encoded_labels(self) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
        label_encoder = LabelEncoder()
        y_train_raw = np.asarray(self.feature_set.y_train)
        y_test_raw = np.asarray(self.feature_set.y_test)
        #label_encoder.fit(np.concatenate([y_train_raw, y_test_raw]))
        label_encoder.fit(y_train_raw)
        self.label_encoder = label_encoder
        return label_encoder.transform(y_train_raw), label_encoder.transform(y_test_raw), label_encoder

    def _decode_labels(self, labels: np.ndarray) -> np.ndarray:
        if self.label_encoder is None:
            return np.asarray(labels)
        return self.label_encoder.inverse_transform(np.asarray(labels))

    def _baseline_models(self) -> Dict[str, object]:
        return {
            "ExtraTrees": ExtraTreesClassifier(
                random_state=RANDOM_STATE, n_jobs=self.model_n_jobs
            ),
            "RandomForest": RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=self.model_n_jobs
            ),
            "XGBoost": xgb.XGBClassifier(
                random_state=RANDOM_STATE, n_jobs=self.model_n_jobs, tree_method="hist"
            ),
            "SVM": SVC(
                random_state=RANDOM_STATE,
                probability=True,
            ),
            "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "LogisticRegression": LogisticRegression(
                max_iter=2000, n_jobs=self.model_n_jobs, random_state=RANDOM_STATE, solver="saga"
            ),
        }

    def train_baselines(self) -> Dict[str, ModelResult]:
        X_train = self.feature_set.X_train_scaled
        y_train, y_test, label_encoder = self._encoded_labels()
        X_test = self.feature_set.X_test_scaled
        baseline_results: Dict[str, ModelResult] = {}

        for name, model in self._baseline_models().items():
            logger.info("Training baseline model: %s", name)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            metrics = compute_metrics(y_test, preds, proba, n_classes=len(label_encoder.classes_))
            result = ModelResult(
                name=name,
                model=model,
                metrics=metrics,
                y_true=self._decode_labels(y_test),
                y_pred=self._decode_labels(preds),
                y_proba=proba,
            )
            baseline_results[name] = result
            self.models[name] = result
            self._persist_model(model, f"{name}_baseline.pkl")
        return baseline_results

    def _objective(self, model_name: str, trial: optuna.Trial) -> float:
        X_train = self.feature_set.X_train_scaled
        y_train, _, _ = self._encoded_labels()
        min_class_count = int(np.min(np.bincount(y_train)))
        n_splits = min(self.cv_folds, min_class_count)
        if n_splits < 2:
            raise ValueError(
                f"Not enough samples in the smallest class for cross-validation: min_class_count={min_class_count}"
            )
        if n_splits < self.cv_folds:
            logger.info("Reducing cv folds from %s to %s based on smallest class count", self.cv_folds, n_splits)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        if model_name == "ExtraTrees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
                "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 16, 24, 32, 48]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.3, 0.5, 0.7]),
                "n_jobs": self.model_n_jobs,
                "random_state": RANDOM_STATE,
            }
            model = ExtraTreesClassifier(**params)
        elif model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
                "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 16, 24, 32, 48]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.3, 0.5, 0.7]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "n_jobs": self.model_n_jobs,
                "random_state": RANDOM_STATE,
            }
            model = RandomForestClassifier(**params)
        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=100),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 10.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
                "n_jobs": self.model_n_jobs,
                "random_state": RANDOM_STATE,
                "tree_method": "hist",
            }
            model = xgb.XGBClassifier(**params)
        elif model_name == "SVM":
            params = {
                "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
                "probability": True,
                "random_state": RANDOM_STATE,
            }
            if params["kernel"] != "linear":
                params["gamma"] = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
            if params["kernel"] == "poly":
                params["degree"] = trial.suggest_int("degree", 2, 5)
                params["coef0"] = trial.suggest_float("coef0", 0.0, 1.0)
            if params["kernel"] == "sigmoid":
                params["coef0"] = trial.suggest_float("coef0", 0.0, 1.0)
            model = SVC(**params)
        elif model_name == "GradientBoosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 800, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.3, 0.5, 0.7]),
                "random_state": RANDOM_STATE,
            }
            model = GradientBoostingClassifier(**params)
        elif model_name == "LogisticRegression":
            penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else None
            params = {
                "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
                "penalty": penalty,
                "l1_ratio": l1_ratio,
                "solver": "saga",
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
                "max_iter": 2000,
                "n_jobs": self.model_n_jobs,
                "random_state": RANDOM_STATE,
            }
            model = LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model for Optuna: {model_name}")

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=self.model_n_jobs,
        )
        return float(np.mean(scores))

    def optimize_models(
        self,
        model_names: Optional[List[str]] = None,
        n_trials: int = 30,
        parallel_trials: int = 4,
    ) -> Dict[str, ModelResult]:
        names = model_names or list(self._baseline_models().keys())
        X_test = self.feature_set.X_test_scaled
        y_train, y_test, label_encoder = self._encoded_labels()
        n_classes = len(label_encoder.classes_)
        tuned_results: Dict[str, ModelResult] = {}

        for name in names:
            logger.info("Optuna tuning for model: %s (trials=%s, parallel=%s)", name, n_trials, parallel_trials)
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._objective(name, trial),
                n_trials=n_trials,
                n_jobs=parallel_trials,
                show_progress_bar=False,
            )

            best_params = study.best_params
            tuned_model = self._build_final_model(name, best_params)
            tuned_model.fit(self.feature_set.X_train_scaled, y_train)

            preds = tuned_model.predict(X_test)
            proba = tuned_model.predict_proba(X_test) if hasattr(tuned_model, "predict_proba") else None
            metrics = compute_metrics(y_test, preds, proba, n_classes=n_classes)

            result = ModelResult(
                name=f"{name}_tuned",
                model=tuned_model,
                metrics={**metrics, "best_cv_score": study.best_value},
                best_params=best_params,
                y_true=self._decode_labels(y_test),
                y_pred=self._decode_labels(preds),
                y_proba=proba,
            )
            self.models[result.name] = result
            tuned_results[result.name] = result
            self._persist_model(tuned_model, f"{name}_tuned.pkl")
            logger.info("Completed tuning for %s; best score=%.4f", name, study.best_value)
        return tuned_results

    def _build_final_model(self, model_name: str, params: Dict) -> object:
        if model_name == "ExtraTrees":
            final_params = {**params, "n_jobs": self.model_n_jobs, "random_state": RANDOM_STATE}
            return ExtraTreesClassifier(**final_params)
        if model_name == "RandomForest":
            final_params = {**params, "n_jobs": self.model_n_jobs, "random_state": RANDOM_STATE}
            return RandomForestClassifier(**final_params)
        if model_name == "XGBoost":
            final_params = {**params, "n_jobs": self.model_n_jobs, "random_state": RANDOM_STATE, "tree_method": "hist"}
            return xgb.XGBClassifier(**final_params)
        if model_name == "SVM":
            final_params = {**params, "probability": True, "random_state": RANDOM_STATE}
            return SVC(**final_params)
        if model_name == "GradientBoosting":
            final_params = {**params, "random_state": RANDOM_STATE}
            return GradientBoostingClassifier(**final_params)
        if model_name == "LogisticRegression":
            lr_params = {**params}
            # Align with the solver used during Optuna search to avoid incompatible defaults.
            lr_params.update({"solver": "saga", "max_iter": 2000, "n_jobs": self.model_n_jobs, "random_state": RANDOM_STATE})
            if lr_params.get("penalty") != "elasticnet":
                lr_params.pop("l1_ratio", None)
            return LogisticRegression(**lr_params)
        raise ValueError(f"Unknown model {model_name}")

    def _persist_model(self, model: object, filename: str) -> None:
        os.makedirs(os.path.join(self.results_dir, "models"), exist_ok=True)
        path = os.path.join(self.results_dir, "models", filename)
        joblib.dump(model, path)
