from pathlib import Path
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_DIR = PROJECT_ROOT / "tourism_project" / "deployment"
TARGET = "ProdTaken"

X_train = pd.read_csv(PROJECT_ROOT / "Xtrain.csv")
X_test = pd.read_csv(PROJECT_ROOT / "Xtest.csv")
y_train = pd.read_csv(PROJECT_ROOT / "ytrain.csv")[TARGET]
y_test = pd.read_csv(PROJECT_ROOT / "ytest.csv")[TARGET]

numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1)),
])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.10],
    "model__subsample": [0.8, 1.0],
}

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file:{PROJECT_ROOT / 'mlruns'}")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("tourism-package-prediction")

with mlflow.start_run():
    search = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    probabilities = search.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_precision": precision_score(y_test, predictions, zero_division=0),
        "test_recall": recall_score(y_test, predictions, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, probabilities),
    }
    mlflow.log_params(search.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(search.best_estimator_, "model")

DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
model_path = DEPLOYMENT_DIR / "tourism_purchase_model.joblib"
joblib.dump(search.best_estimator_, model_path)

print("Best parameters:", search.best_params_)
print("Metrics:", metrics)
print(classification_report(y_test, predictions, digits=3))
print(f"Saved model to {model_path}")
