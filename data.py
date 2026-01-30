from config import Config
from preprocessing import prepare_for_classification, to_numpy


def dataset(name: str | None = None, config: Config | None = None):
    config = config or Config()
    name = name or config.dataset_name
    ds = prepare_for_classification(name, config=config, seed=config.seed)
    X_train, y_train = to_numpy(ds["train"])
    X_val, y_val = to_numpy(ds["val"]) if "val" in ds else (None, None)
    X_test, y_test = to_numpy(ds["test"])
    print("train:", X_train.shape, y_train.shape)
    if X_val is not None:
        print("val:", X_val.shape, y_val.shape)
    print("test:", X_test.shape, y_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    dataset()
