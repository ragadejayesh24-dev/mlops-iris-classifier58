
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "iris_features.csv"
ARTIFACT_DIR = PROJECT_ROOT / "mlflow_outputs"
ARTIFACT_DIR.mkdir(exist_ok=True)

# MLflow experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-classification-baseline")

# Load dataset
df = pd.read_csv(DATA_PATH)

feature_columns = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

target_column = "species"

X = df[feature_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Three model configurations
models = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1000),
        {"model_type": "LogisticRegression", "max_iter": 1000},
    ),
    (
        "Shallow Random Forest",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42,
        ),
        {
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 3,
        },
    ),
    (
        "Deep Random Forest",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
        ),
        {
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": "None",
        },
    ),
]

for run_name, model, parameters in models:
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision_macro": precision_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
        }

        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

        # Save and log confusion matrix
        cm = confusion_matrix(y_test, predictions)
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=model.classes_,
        )
        display.plot()
        plt.title(run_name)
        plt.tight_layout()

        image_path = ARTIFACT_DIR / (
            run_name.lower().replace(" ", "_") + "_confusion_matrix.png"
        )
        plt.savefig(image_path)
        plt.close()

        mlflow.log_artifact(str(image_path))

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"\nRun: {run_name}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

print("\nAll three MLflow experiments completed.")
