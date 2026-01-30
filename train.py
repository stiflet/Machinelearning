import time

from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

from config import Config
from data import dataset
from models import build_boosting, build_random_forest, build_svm, build_voting


def _with_verbose(params: dict, *, enabled: bool, value):
    if not enabled:
        return params
    if "verbose" in params:
        return params
    updated = dict(params)
    updated["verbose"] = value
    return updated


def build_model(config: Config, model_name: str):
    if model_name == "svm":
        params = _with_verbose(config.svm_params, enabled=config.training_verbose, value=True)
        return build_svm(**params)
    if model_name == "random_forest":
        params = _with_verbose(config.random_forest_params, enabled=config.training_verbose, value=1)
        return build_random_forest(**params)
    if model_name == "boosting":
        params = _with_verbose(config.boosting_params, enabled=config.training_verbose, value=1)
        return build_boosting(**params)
    if model_name == "voting":
        voting_params = _with_verbose(config.voting_params, enabled=config.training_verbose, value=1)
        return build_voting(
            svm_params=config.svm_params,
            random_forest_params=config.random_forest_params,
            **voting_params,
        )
    raise ValueError(f"Unknown model: {model_name}")


def train_and_eval(config: Config | None = None):
    config = config or Config()
    print("loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = dataset(config=config)
    print("data loaded")
    print(f"train size: {X_train.shape[0]}")
    if X_val is not None:
        print(f"val size: {X_val.shape[0]}")
    print(f"test size: {X_test.shape[0]}")

    model_list = config.model if isinstance(config.model, list) else [config.model]
    trained = {}
    val_scores: dict[str, float] = {}
    total_models = len(model_list)
    for idx, model_name in enumerate(model_list, start=1):
        model = build_model(config, model_name)
        print(f"model {idx}/{total_models}: {model_name}")
        print(f"model params ({model_name}): {model.get_params()}")
        print(f"training starting ({model_name})...")
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        print("training finished")
        print(f"training time ({model_name}): {elapsed:.2f}s")

        if X_val is not None:
            val_pred = model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_scores[model_name] = val_acc
            print("val accuracy:", val_acc)

        model_path = f"model_{model_name}.pth"
        dump(model, model_path)
        print(f"model saved: {model_path}")
        trained[model_name] = model

    if X_val is not None and val_scores:
        best_model_name = max(val_scores, key=val_scores.get)
        print(f"best model by val accuracy: {best_model_name} ({val_scores[best_model_name]:.4f})")
    else:
        best_model_name = model_list[0]
        print("no validation set available; evaluating test on first model only")

    best_model = trained[best_model_name]
    print(f"evaluating test set ({best_model_name})...")
    test_pred = best_model.predict(X_test)
    print("test accuracy:", accuracy_score(y_test, test_pred))
    print(classification_report(y_test, test_pred))

    return trained


if __name__ == "__main__":
    train_and_eval()
