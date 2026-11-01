
import pandas as pd
import pytest

from src.pipeline.validate import validate, DataValidationError


@pytest.fixture
def valid_df():
    return pd.DataFrame({
        "sepal length (cm)": [5.1, 4.9, 4.7],
        "sepal width (cm)": [3.5, 3.0, 3.2],
        "petal length (cm)": [1.4, 1.4, 1.3],
        "petal width (cm)": [0.2, 0.2, 0.2],
        "species": ["setosa", "versicolor", "virginica"],
        "sepal_area": [17.85, 14.70, 15.04],
        "petal_area": [0.28, 0.28, 0.26],
        "sepal_to_petal_length_ratio": [
            3.64, 3.50, 3.62
        ],
        "petal_length_bin": ["small", "small", "small"],
    })


def test_validate_accepts_valid_data(tmp_path, valid_df):
    input_path = tmp_path / "valid.csv"
    valid_df.to_csv(input_path, index=False)

    result = validate(str(input_path))

    assert len(result) == 3
    assert len(result.columns) == 9


def test_validate_rejects_missing_columns(tmp_path, valid_df):
    df = valid_df.drop(columns=["species"])

    input_path = tmp_path / "missing.csv"
    df.to_csv(input_path, index=False)

    with pytest.raises(DataValidationError):
        validate(str(input_path))


def test_validate_rejects_missing_values(tmp_path, valid_df):
    df = valid_df.copy()
    df.loc[0, "species"] = None

    input_path = tmp_path / "missing_values.csv"
    df.to_csv(input_path, index=False)

    with pytest.raises(DataValidationError):
        validate(str(input_path))


def test_validate_rejects_invalid_species(tmp_path, valid_df):
    df = valid_df.copy()
    df.loc[0, "species"] = "unknown"

    input_path = tmp_path / "invalid_species.csv"
    df.to_csv(input_path, index=False)

    with pytest.raises(DataValidationError):
        validate(str(input_path))
