from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from gen_class.config import FIGURE_DPI, PLOT_STYLE
from gen_class.modeling import ModelResult
from gen_class.unsupervised import ClusteringResult, ProjectionResult


class ResultsSaver:
    """Persist metrics, embeddings, and plots into the configured results directory."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, "plots")
        self.reports_dir = os.path.join(results_dir, "reports")
        self.data_dir = os.path.join(results_dir, "data_stats")
        for path in (self.plots_dir, self.reports_dir, self.data_dir):
            os.makedirs(path, exist_ok=True)
        try:
            plt.style.use(PLOT_STYLE)
        except OSError:
            plt.style.use("default")
        sns.set_theme()

    def save_model_results(
        self,
        baseline_results: Dict[str, ModelResult],
        tuned_results: Dict[str, ModelResult],
        class_names: Optional[Iterable[str]] = None,
    ) -> None:
        combined = {**baseline_results, **tuned_results}
        if not combined:
            return

        serialized: Dict[str, dict] = {}
        summary_rows = []
        for name, res in combined.items():
            metrics = res.metrics or {}
            entry: Dict[str, object] = {"metrics": metrics}
            if res.best_params:
                entry["best_params"] = res.best_params
            if res.y_true is not None and res.y_pred is not None:
                try:
                    entry["classification_report"] = classification_report(
                        res.y_true,
                        res.y_pred,
                        target_names=list(class_names) if class_names else None,
                        output_dict=True,
                        zero_division=0,
                    )
                    entry["confusion_matrix"] = confusion_matrix(res.y_true, res.y_pred).tolist()
                except Exception:
                    # Graceful fallback if metrics cannot be computed
                    pass
            serialized[name] = entry
            summary_rows.append({"model": name, **metrics})

        with open(os.path.join(self.reports_dir, "model_results.json"), "w") as f:
            json.dump(serialized, f, indent=2)

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(self.reports_dir, "model_metrics.csv"), index=False
            )

    def save_confusion_matrices(
        self,
        baseline_results: Dict[str, ModelResult],
        tuned_results: Dict[str, ModelResult],
        class_names: Optional[Iterable[str]] = None,
    ) -> None:
        combined = {**baseline_results, **tuned_results}
        if not combined:
            return

        for name, res in combined.items():
            if res.y_true is None or res.y_pred is None:
                continue
            y_true = np.asarray(res.y_true)
            y_pred = np.asarray(res.y_pred)
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            labels = (
                [str(cn) for cn in class_names]
                if class_names and len(class_names) >= len(unique_labels)
                else [str(lbl) for lbl in unique_labels]
            )

            cm = confusion_matrix(y_true, y_pred, labels=unique_labels, normalize="true")
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix - {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f"{name}_confusion_matrix.png"), dpi=FIGURE_DPI)
            plt.close()

    def save_feature_metadata(self, feature_set, loaded_data) -> None:
        def _to_str_list(values):
            return [str(v) for v in values] if values is not None else []

        payload = {
            "selected_feature_count": len(getattr(feature_set, "selected_features", [])),
            "selected_features": _to_str_list(getattr(feature_set, "selected_features", [])),
            "train_size": int(len(getattr(feature_set, "y_train", []))),
            "test_size": int(len(getattr(feature_set, "y_test", []))),
        }
        if loaded_data is not None:
            class_names = _to_str_list(getattr(loaded_data, "class_names", []))
            payload["class_names"] = class_names
            payload["label_to_index"] = {
                name: int(idx) for idx, name in enumerate(class_names)
            }

        with open(os.path.join(self.data_dir, "feature_metadata.json"), "w") as f:
            json.dump(payload, f, indent=2)

    def save_projections(self, projections: Dict[str, ProjectionResult]) -> None:
        meta: Dict[str, dict] = {}
        for key, res in projections.items():
            np.save(os.path.join(self.data_dir, f"{key}_embedding.npy"), res.embedding)
            meta[key] = {
                "name": res.name,
                "shape": list(res.embedding.shape),
                "extra": res.extra,
            }
        with open(os.path.join(self.data_dir, "projections.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def save_unsupervised_results(self, clusters: Dict[str, ClusteringResult]) -> None:
        if not clusters:
            return
        serialized: Dict[str, dict] = {}
        for name, res in clusters.items():
            serialized[name] = {
                "ari": res.ari,
                "silhouette": res.silhouette,
                "extra": res.extra,
                "n_unique_labels": int(len(np.unique(res.labels))),
            }
            np.save(os.path.join(self.data_dir, f"{name}_labels.npy"), res.labels)
        with open(os.path.join(self.reports_dir, "unsupervised_results.json"), "w") as f:
            json.dump(serialized, f, indent=2)

    def save_projection_plots(
        self,
        projections: Dict[str, ProjectionResult],
        labels: np.ndarray,
        class_names: Optional[Iterable[str]] = None,
    ) -> None:
        if projections is None or labels is None:
            return

        labels = np.asarray(labels)
        label_lookup = {
            int(idx): (class_names[idx] if class_names and idx < len(class_names) else str(idx))
            for idx in np.unique(labels)
        }
        label_names = [label_lookup[int(lbl)] for lbl in labels]

        for key, res in projections.items():
            emb = res.embedding
            plt.figure(figsize=(6, 5))
            sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=label_names, palette="tab10", s=30, linewidth=0)
            plt.title(f"{res.name} embedding")
            plt.legend(title="label", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f"{key}_embedding.png"), dpi=FIGURE_DPI)
            plt.close()

    def save_clustering_plot(
        self,
        projections: Dict[str, ProjectionResult],
        clusters: Dict[str, ClusteringResult],
        labels: np.ndarray,
        class_names: Optional[Iterable[str]] = None,
    ) -> None:
        if projections is None or clusters is None or "pca" not in projections:
            return

        pca_emb = projections["pca"].embedding
        y = np.asarray(labels)
        label_lookup = {
            int(idx): (class_names[idx] if class_names and idx < len(class_names) else str(idx))
            for idx in np.unique(y)
        }
        y_names = [label_lookup[int(lbl)] for lbl in y]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        sns.scatterplot(ax=axes[0], x=pca_emb[:, 0], y=pca_emb[:, 1], hue=y_names, palette="tab10", s=30, linewidth=0)
        axes[0].set_title("PCA: true labels")
        axes[0].legend(title="label", bbox_to_anchor=(1.05, 1), loc="upper left")

        if "GMM" in clusters:
            sns.scatterplot(
                ax=axes[1],
                x=pca_emb[:, 0],
                y=pca_emb[:, 1],
                hue=clusters["GMM"].labels,
                palette="viridis",
                s=30,
                linewidth=0,
            )
            axes[1].set_title("GMM clusters")
            axes[1].legend(title="cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            axes[1].text(0.5, 0.5, "GMM not available", ha="center", va="center")
            axes[1].set_axis_off()

        if "DBSCAN" in clusters:
            sns.scatterplot(
                ax=axes[2],
                x=pca_emb[:, 0],
                y=pca_emb[:, 1],
                hue=clusters["DBSCAN"].labels,
                palette="tab20",
                s=30,
                linewidth=0,
            )
            eps_val = clusters["DBSCAN"].extra["eps"] if clusters["DBSCAN"].extra else None
            axes[2].set_title(f"DBSCAN clusters (eps={eps_val})")
            axes[2].legend(title="cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            axes[2].text(0.5, 0.5, "DBSCAN not available", ha="center", va="center")
            axes[2].set_axis_off()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "clustering_overview.png"), dpi=FIGURE_DPI)
        plt.close()

    def save_all(self, outputs) -> None:
        """Persist all pipeline artifacts derived from PipelineOutputs."""
        class_names = getattr(getattr(outputs, "data", None), "class_names", None)
        self.save_model_results(outputs.baseline_results, outputs.tuned_results, class_names=class_names)
        self.save_confusion_matrices(outputs.baseline_results, outputs.tuned_results, class_names=class_names)
        self.save_feature_metadata(outputs.features, getattr(outputs, "data", None))
        self.save_projections(outputs.projections)
        self.save_unsupervised_results(outputs.unsupervised_results)
        self.save_projection_plots(outputs.projections, outputs.features.y_train, class_names=class_names)
        self.save_clustering_plot(outputs.projections, outputs.unsupervised_results, outputs.features.y_train, class_names=class_names)
