
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


@pytest.fixture
def model_data():
    return pd.DataFrame({
        "sepal length (cm)": [
            5.1, 4.9, 4.7, 5.0, 5.4, 6.0,
            6.1, 6.2, 6.5, 6.7, 6.3, 6.4,
            5.5, 5.7, 5.8, 6.0, 6.2, 6.4
        ],
        "sepal width (cm)": [
            3.5, 3.0, 3.2, 3.6, 3.9, 2.2,
            2.8, 2.9, 3.0, 3.1, 2.9, 3.0,
            2.4, 2.5, 2.6, 2.7, 2.8, 2.9
        ],
        "petal length (cm)": [
            1.4, 1.4, 1.3, 1.4, 1.7, 4.0,
            4.5, 4.3, 5.5, 5.6, 5.1, 5.3,
            3.8, 4.0, 4.1, 4.5, 4.8, 5.0
        ],
        "petal width (cm)": [
            0.2, 0.2, 0.2, 0.2, 0.4, 1.0,
            1.3, 1.3, 2.0, 2.1, 1.8, 1.9,
            1.1, 1.2, 1.3, 1.5, 1.6, 1.7
        ],
        "species": [
            "setosa", "setosa", "setosa",
            "setosa", "setosa", "versicolor",
            "versicolor", "versicolor", "virginica",
            "virginica", "virginica", "virginica",
            "versicolor", "versicolor", "versicolor",
            "versicolor", "versicolor", "virginica"
        ]
    })


def test_model_accuracy(model_data):
    X = model_data.drop(columns=["species"])
    y = model_data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.70


def test_model_f1_score(model_data):
    X = model_data.drop(columns=["species"])
    y = model_data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = f1_score(
        y_test, predictions, average="macro"
    )

    assert score >= 0.70


def test_model_predicts_valid_classes(model_data):
    X = model_data.drop(columns=["species"])
    y = model_data["species"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)
    predictions = model.predict(X)

    valid_classes = set(y.unique())

    assert set(predictions).issubset(valid_classes)


def test_model_prediction_count(model_data):
    X = model_data.drop(columns=["species"])
    y = model_data["species"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
