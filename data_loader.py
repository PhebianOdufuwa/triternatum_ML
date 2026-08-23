from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from .config import DEFAULT_IMG_SIZE, IMG_MEAN, IMG_STD, alphabet, char_to_idx
from .read_data import build_sequence_dataframe

logger = logging.getLogger(__name__)


def encode_seq(seq: str) -> np.ndarray:
    """One-hot encode a genomic sequence with graceful fallback for unknown bases (no padding)."""
    seq = seq.strip().upper()
    L = len(seq)
    mat = np.zeros((L, len(alphabet)), dtype=np.uint8)
    for i, ch in enumerate(seq):
        mat[i, char_to_idx.get(ch, char_to_idx["A"])] = 1
    return mat.flatten()


def preprocess_image(img_path: str, target_size: tuple[int, int] = DEFAULT_IMG_SIZE) -> tf.Tensor:
    """Load and standardize an image; return zeros if the file cannot be read."""
    try:
        img_raw = tf.io.read_file(img_path)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        img_tensor = tf.image.resize(img_tensor, target_size)
        img_tensor = img_tensor / 255.0
        img_tensor = (img_tensor - IMG_MEAN) / IMG_STD
        return img_tensor
    except Exception as exc: 
        print(f"[data_loader] Warning: failed to read {img_path}: {exc}")
        return tf.zeros((target_size[0], target_size[1], 3), dtype=tf.float32)


@dataclass
class LoadedData:
    raw_df: pd.DataFrame
    genomic_matrix: np.ndarray
    images: tf.Tensor
    labels: np.ndarray
    label_encoder: Optional[LabelEncoder]
    class_names: List[str]
    img_paths: List[str]


class DataLoader:
    """Prepare the merged genomic + image dataset using the curated read_data pipeline."""

    def __init__(self, image_size: tuple[int, int] = DEFAULT_IMG_SIZE, use_label_encoder: bool = False):
        self.image_size = image_size
        self.use_label_encoder = use_label_encoder
        self.loaded: Optional[LoadedData] = None

    def load(
        self,
        fasta_path: str,
        mapping_csv_path: str,
        img_prefix: str,
        label_map: Optional[dict[str, int]] = None,
        use_label_encoder: Optional[bool] = None,
    ) -> LoadedData:
        """
        Build the dataset directly from FASTA and mapping files using read_data.build_sequence_dataframe.
        """
        logger.info("Loading dataset using read_data.build_sequence_dataframe")
        df = build_sequence_dataframe(
            fasta_path=fasta_path,
            mapping_csv_path=mapping_csv_path,
            img_prefix=img_prefix,
            label_map=label_map,
        )

        required_cols = {"seqs", "img", "label"}
        missing = required_cols.difference(df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Dataset is missing required columns: {missing_list}")

        # Drop rows with missing critical fields and missing image files
        df["img"] = df["img"].apply(lambda p: os.path.abspath(p))
        mask_complete = df["seqs"].notna() & df["label"].notna()
        mask_img_exists = df["img"].apply(os.path.exists)
        filtered = df[mask_complete & mask_img_exists].reset_index(drop=True)

        dropped_missing = len(df) - len(filtered)
        if dropped_missing > 0:
            logger.warning("Dropped %s rows due to missing fields or image files", dropped_missing)

        if filtered.empty:
            raise ValueError("No samples left after filtering for complete rows with existing images.")

        df = filtered

        # Encode genomic sequences
        logger.info("Encoding %s genomic sequences", len(df))
        seq_lengths = df["seqs"].str.len()
        unique_lengths = seq_lengths.unique()
        if len(unique_lengths) > 1:
            raise ValueError(
                f"Sequence lengths are inconsistent (counts: {seq_lengths.value_counts().to_dict()}). "
                "Please align lengths or re-enable padding if variable lengths are expected."
            )
        genomic_matrix = np.vstack([encode_seq(seq) for seq in df["seqs"]])

        # Resolve absolute image paths relative to the provided prefix
        img_paths = df["img"].tolist()

        # Preprocess images
        logger.info("Preprocessing %s images", len(img_paths))
        images = tf.stack([preprocess_image(path, self.image_size) for path in img_paths])

        # Encode labels optionally. Default behavior keeps user-defined label IDs as-is.
        encode_labels = self.use_label_encoder if use_label_encoder is None else use_label_encoder
        if encode_labels:
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(df["label"])
            class_names = [str(c) for c in label_encoder.classes_]
        else:
            label_encoder = LabelEncoder()
            label_encoder.fit(df["label"])
            labels = df["label"].astype(int).to_numpy()
            class_names = [str(c) for c in sorted(df["label"].unique())]

        self.loaded = LoadedData(
            raw_df=df,
            genomic_matrix=genomic_matrix,
            images=images,
            labels=labels,
            label_encoder=label_encoder,
            class_names=class_names,
            img_paths=img_paths,
        )
        return self.loaded
