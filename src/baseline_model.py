
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("data/processed/iris_features.csv")

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

encoder = LabelEncoder()
y = encoder.fit_transform(df["species"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Baseline Decision Tree
model = DecisionTreeClassifier(random_state=42)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="f1_macro",
)

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# Log results in MLflow
mlflow.set_experiment("iris-hyperparameter-tuning")

with mlflow.start_run(run_name="baseline_decision_tree"):
    mlflow.log_param("model", "DecisionTreeClassifier")
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("cv_f1_macro_mean", scores.mean())
    mlflow.log_metric("cv_f1_macro_std", scores.std())
    mlflow.log_metric("test_accuracy", accuracy)

print(f"Baseline CV F1 Macro: {scores.mean():.4f}")
print(f"Baseline CV F1 Std: {scores.std():.4f}")
print(f"Baseline test accuracy: {accuracy:.4f}")
print("Baseline results saved to MLflow.")
