from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from skimage.feature import hog
from config import Config


def load_hf_dataset(name: str, *, split: Optional[str] = None, cache_dir: Optional[str] = None):
    return load_dataset(name, split=split, cache_dir=cache_dir)


def ensure_splits(
    dataset,
    *,
    config: Optional[Config] = None,
    seed: int = 42,
) -> DatasetDict:
    if isinstance(dataset, DatasetDict):
        if "train" in dataset and ("test" not in dataset or "val" not in dataset):
            return ensure_splits(dataset["train"], config=config, seed=seed)
        return dataset

    config = config or Config()
    test_size = config.test_size
    val_size = config.val_size
    holdout = test_size + val_size
    if holdout <= 0 or holdout >= 1:
        raise ValueError("test_size + val_size must be between 0 and 1")

    split = dataset.train_test_split(test_size=holdout, seed=seed)
    if val_size <= 0:
        return DatasetDict(train=split["train"], test=split["test"])

    val_ratio = val_size / holdout
    test_val = split["test"].train_test_split(test_size=val_ratio, seed=seed)
    return DatasetDict(train=split["train"], val=test_val["test"], test=test_val["train"])


def _image_to_array(image, *, resize: Optional[Tuple[int, int]], normalize: bool) -> np.ndarray:
    if resize is not None:
        image = image.resize(resize)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if normalize:
        arr = arr.astype("float32") / 255.0
    return arr


def _image_to_hog(image, *, resize: Optional[Tuple[int, int]], normalize: bool, hog_params: dict) -> np.ndarray:
    if resize is not None:
        image = image.resize(resize)
    if getattr(image, "mode", None) != "L":
        image = image.convert("L")
    arr = np.asarray(image)
    if normalize:
        arr = arr.astype("float32") / 255.0
    return hog(arr, **hog_params)


def flatten_images(
    dataset,
    *,
    image_col: str = "image",
    label_col: str = "label",
    resize: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    data_format: str = "flatten",
    hog_params: Optional[dict] = None,
) -> DatasetDict | Dataset:
    hog_params = hog_params or {
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
    }

    def _prepare_batch(batch):
        images = batch[image_col]
        labels = batch[label_col]
        if data_format == "flatten":
            features = [
                _image_to_array(img, resize=resize, normalize=normalize).reshape(-1)
                for img in images
            ]
        elif data_format == "hog":
            features = [
                _image_to_hog(img, resize=resize, normalize=normalize, hog_params=hog_params)
                for img in images
            ]
        else:
            raise ValueError(f"Unknown data_format: {data_format}")
        return {"features": features, "labels": labels}

    if isinstance(dataset, DatasetDict):
        remove_columns = dataset["train"].column_names
        return dataset.map(_prepare_batch, batched=True, remove_columns=remove_columns)
    return dataset.map(_prepare_batch, batched=True, remove_columns=dataset.column_names)


def to_numpy(
    dataset: Dataset,
    *,
    feature_col: str = "features",
    label_col: str = "labels",
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.stack(dataset[feature_col])
    y = np.asarray(dataset[label_col])
    return X, y


def prepare_for_classification(
    name: str,
    *,
    image_col: str = "image",
    label_col: str = "label",
    resize: Optional[Tuple[int, int]] = None,
    normalize: Optional[bool] = None,
    data_format: Optional[str] = None,
    hog_params: Optional[dict] = None,
    config: Optional[Config] = None,
    seed: int = 42,
):
    dataset = load_hf_dataset(name)
    dataset = ensure_splits(dataset, config=config, seed=seed)
    config = config or Config()
    data_format = data_format or config.data_format
    hog_params = hog_params or config.hog_params
    if resize is None:
        resize = config.image_size
    if normalize is None:
        normalize = config.normalize
    dataset = flatten_images(
        dataset,
        image_col=image_col,
        label_col=label_col,
        resize=resize,
        normalize=normalize,
        data_format=data_format,
        hog_params=hog_params,
    )
    return dataset
