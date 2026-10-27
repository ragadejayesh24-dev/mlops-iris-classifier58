
import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import randint

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
y = LabelEncoder().fit_transform(df["species"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Parameters for Random Search
param_distributions = {
    "n_estimators": randint(50, 201),
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2", None],
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=30,
    cv=5,
    scoring="f1_macro",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("Starting Random Search. Please wait...")
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, predictions)
test_f1 = f1_score(y_test, predictions, average="macro")

mlflow.set_experiment("iris-hyperparameter-tuning")

with mlflow.start_run(run_name="random_search_random_forest"):
    mlflow.log_param("search_method", "RandomizedSearchCV")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("iterations", 30)
    mlflow.log_params(random_search.best_params_)
    mlflow.log_metric("best_cv_f1_macro", random_search.best_score_)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1_macro", test_f1)
    mlflow.log_metric("configurations_tested", len(random_search.cv_results_))

print("\nRandom Search completed!")
print("Best parameters:", random_search.best_params_)
print(f"Best CV F1 Macro: {random_search.best_score_:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test F1 Macro: {test_f1:.4f}")
print("Random Search results saved to MLflow.")
