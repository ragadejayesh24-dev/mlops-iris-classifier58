
import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# Load the dataset
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
y = LabelEncoder().fit_transform(df["species"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Grid of parameters: 3 x 3 x 2 x 2 x 2 = 72 combinations
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", None],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
)

print("Starting Grid Search. Please wait...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, predictions)
test_f1 = f1_score(y_test, predictions, average="macro")

mlflow.set_experiment("iris-hyperparameter-tuning")

with mlflow.start_run(run_name="grid_search_random_forest"):
    mlflow.log_param("search_method", "GridSearchCV")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1_macro", grid_search.best_score_)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1_macro", test_f1)
    mlflow.log_metric("combinations_tested", len(grid_search.cv_results_))

print("\nGrid Search completed!")
print("Best parameters:", grid_search.best_params_)
print(f"Best CV F1 Macro: {grid_search.best_score_:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test F1 Macro: {test_f1:.4f}")
print("Grid Search results saved to MLflow.")
