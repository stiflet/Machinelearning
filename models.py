from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from config import Config

config = Config()

def build_svm(**kwargs):
    return SVC(**kwargs)


def build_random_forest(**kwargs):
    return RandomForestClassifier(**kwargs)


def build_boosting(**kwargs):
    return GradientBoostingClassifier(**kwargs)


def build_voting(
    *,
    svm_params: dict,
    random_forest_params: dict,
    voting: str = "soft",
    weights=None,
    **kwargs,
):
    if voting not in {"hard", "soft"}:
        raise ValueError("voting must be 'hard' or 'soft'")
    if weights is not None and len(weights) != 2:
        raise ValueError("weights must be length 2 for (svm, random_forest)")
    svm_params = dict(svm_params)
    if voting == "soft" and "probability" not in svm_params:
        svm_params["probability"] = True
    svm_model = build_svm(**svm_params)
    rf_model = build_random_forest(**random_forest_params)
    return VotingClassifier(
        estimators=[("svm", svm_model), ("random_forest", rf_model)],
        voting=voting,
        weights=weights,
        **kwargs,
    )


def get_models():
    return {
        "svm": build_svm(),
        "random_forest": build_random_forest(),
        "boosting": build_boosting(),
        "voting": build_voting(svm_params=config.svm_params, random_forest_params=config.random_forest_params, voting=config.voting_params.get('voting','soft')),
    }
