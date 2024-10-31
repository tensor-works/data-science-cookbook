import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

load_dotenv()

# Assuming feature_df and targets_df are already defined
data_dir = os.getenv("DATA")
normal_df = pd.read_csv(Path(data_dir) / "normal_dataset.csv")
minmax_df = pd.read_csv(Path(data_dir) / "minmax_dataset.csv")
targets_df = pd.read_csv(Path(data_dir) / "target.csv")

Xn = normal_df.sample(n=40000)
ids = Xn.id
Xn = Xn.drop(columns=["id"]).values
Xmm = minmax_df[minmax_df.id.isin(ids)].drop(columns=["id"]).values
y = targets_df[targets_df.id.isin(ids)].drop(columns=["id"]).values
# Split the data
Xn_train, Xn_test, y_train, y_test = train_test_split(Xn, y, test_size=0.2, random_state=42)
Xmm_train, Xmm_test, y_train, y_test = train_test_split(Xmm, y, test_size=0.2, random_state=42)
from sklearn.metrics import f1_score

# Use ray.put() to store the data in Ray's object store
Xmm_train_id = ray.put(Xmm_train)
Xmm_test_id = ray.put(Xmm_test)
Xn_train_id = ray.put(Xn_train)
Xn_test_id = ray.put(Xn_test)
y_train_id = ray.put(y_train)
y_test_id = ray.put(y_test)


def evaluate_model(y_true, y_pred):
    return f1_score(y_true, y_pred)


def train_logistic_regression(config, X_train, X_test, y_train, y_test):
    clf = LogisticRegression(C=config["C"], max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = evaluate_model(y_test, y_pred)
    return {"f1_score": f1}


def train_random_forest(config, X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=config["n_estimators"],
                                 max_depth=config["max_depth"],
                                 min_samples_split=config["min_samples_split"],
                                 random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = evaluate_model(y_test, y_pred)
    return {"f1_score": f1}


def create_dnn_model(config, input_shape):
    model = Sequential([
        Dense(config["units_1"], activation="relu", input_shape=input_shape),
        Dense(config["units_2"], activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=config["learning_rate"]),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def train_dnn(config, X_train, X_test, y_train, y_test):
    model = create_dnn_model(config, (X_train.shape[1],))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()
    f1 = evaluate_model(y_test, y_pred)
    return {"f1_score": f1}


def train_model(config):
    y_train = ray.get(y_train_id)
    y_test = ray.get(y_test_id)
    if config["model_type"] == "logistic_regression":
        X_train = ray.get(Xmm_train_id)
        X_test = ray.get(Xmm_test_id)
        return train_logistic_regression(config, X_train, X_test, y_train, y_test)
    elif config["model_type"] == "random_forest":
        X_train = ray.get(Xmm_train_id)
        X_test = ray.get(Xmm_test_id)
        return train_random_forest(config, X_train, X_test, y_train, y_test)
    # elif config["model_type"] == "dnn":
    #     X_train = ray.get(Xn_train_id)
    #     X_test = ray.get(Xn_test_id)
    #     return train_dnn(config, X_train, X_test, y_train, y_test)


search_space = {
    "model_type": tune.choice(["logistic_regression", "random_forest"]),  #, "dnn"]),
    # Logistic Regression hyperparameters
    "C": tune.loguniform(1e-4, 1e4),
    # Random Forest hyperparameters
    "n_estimators": tune.randint(10, 200),
    "max_depth": tune.randint(3, 20),
    "min_samples_split": tune.randint(2, 11),
    # DNN hyperparameters
    # "units_1": tune.randint(16, 128),
    # "units_2": tune.randint(16, 128),
    # "learning_rate": tune.loguniform(1e-4, 1e-2)
}

ray.init(num_cpus=8, num_gpus=1, ignore_reinit_error=True)

analysis = tune.run(train_model,
                    config=search_space,
                    num_samples=1000,
                    scheduler=ASHAScheduler(metric="f1_score", mode="max"),
                    resources_per_trial={
                        "cpu": 1,
                        "gpu": 0
                    })

best_config = analysis.get_best_config(metric="f1_score", mode="max")
best_trial = analysis.get_best_trial(metric="f1_score", mode="max")

print("Best config:", best_config)
print("Best F1 Score:", best_trial.last_result["f1_score"])

ray.shutdown()
