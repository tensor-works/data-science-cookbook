import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.inspection import permutation_importance


def evaluate_permuation(X, y, model, n_splits=5, n_repeat=10) -> pd.Series:
    # Create KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    importances = list()

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        result = permutation_importance(model,
                                        X_test,
                                        y_test,
                                        n_repeats=n_repeat,
                                        scoring=make_scorer(f1_score),
                                        random_state=42,
                                        n_jobs=-1)

        importances.append(result.importances_mean)

    importances = np.array(importances)
    mean_importances = np.mean(importances, axis=0)
    feature_importance = pd.Series(mean_importances, index=X.columns)

    return feature_importance
