from dataclasses import dataclass, field


@dataclass
class Config:
    dataset_name: str = "nielsr/eurosat-demo"
    test_size: float = 0.1
    val_size: float = 0.1
    model: list = field(default_factory=lambda: ["svm","random_forest","boosting","voting"])
    seed: int = 42
    data_format: str = "hog"
    image_size: tuple[int, int] = (64, 64)
    normalize: bool = True
    training_verbose: bool = True
    svm_params: dict = field(default_factory=lambda: {"kernel": "rbf", "C": 1.0, "gamma": "scale"})
    random_forest_params: dict = field(
        default_factory=lambda: {"n_estimators": 200, "max_depth": None, "n_jobs": -1, "random_state": 42}
    )
    boosting_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 50,
            "learning_rate": 0.1,
            "max_depth": 2,
            "subsample": 0.7,
            "max_features": "sqrt",
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
            "tol": 1e-4,
            "verbose": 1,
            "random_state": 42,
        }
    )
    voting_params: dict = field(default_factory=lambda: {"voting": "soft", "weights": None, "verbose": 1, "n_jobs": -1})
    test_models: list = field(default_factory=lambda: ["svm","random_forest", "voting"])
    hog_params: dict = field(
        default_factory=lambda: {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2),
            "block_norm": "L2-Hys",
        }
    )
