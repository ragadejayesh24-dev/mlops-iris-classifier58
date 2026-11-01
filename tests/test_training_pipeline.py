
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "sepal length (cm)": [
            5.1, 4.9, 4.7, 6.0, 6.1, 6.2,
            6.5, 6.7, 6.3, 5.5, 5.7, 5.8
        ],
        "sepal width (cm)": [
            3.5, 3.0, 3.2, 2.2, 2.8, 2.9,
            3.0, 3.1, 2.9, 2.4, 2.5, 2.6
        ],
        "petal length (cm)": [
            1.4, 1.4, 1.3, 4.0, 4.5, 4.3,
            5.5, 5.6, 5.1, 3.8, 4.0, 4.1
        ],
        "petal width (cm)": [
            0.2, 0.2, 0.2, 1.0, 1.3, 1.3,
            2.0, 2.1, 1.8, 1.1, 1.2, 1.3
        ],
        "species": [
            "setosa", "setosa", "setosa",
            "versicolor", "versicolor", "versicolor",
            "virginica", "virginica", "virginica",
            "versicolor", "versicolor", "versicolor"
        ]
    })


def test_training_data_split(sample_data):
    X = sample_data.drop(columns=["species"])
    y = sample_data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    assert len(X_train) == 9
    assert len(X_test) == 3
    assert len(y_train) == 9
    assert len(y_test) == 3


def test_model_training(sample_data):
    X = sample_data.drop(columns=["species"])
    y = sample_data["species"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    assert hasattr(model, "predict")
    assert hasattr(model, "classes_")


def test_model_predictions(sample_data):
    X = sample_data.drop(columns=["species"])
    y = sample_data["species"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset(set(y))


def test_model_reproducibility(sample_data):
    X = sample_data.drop(columns=["species"])
    y = sample_data["species"]

    model1 = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model2 = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model1.fit(X, y)
    model2.fit(X, y)

    predictions1 = model1.predict(X)
    predictions2 = model2.predict(X)

    assert (predictions1 == predictions2).all()
