
import pandas as pd
import pytest

from src.pipeline.features import engineer_features


@pytest.fixture
def sample_df(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.1, 4.9, 4.7],
        "sepal width (cm)": [3.5, 3.0, 3.2],
        "petal length (cm)": [1.4, 1.4, 1.3],
        "petal width (cm)": [0.2, 0.2, 0.2],
        "species": ["setosa", "setosa", "setosa"],
    })

    input_path = tmp_path / "input.csv"
    df.to_csv(input_path, index=False)

    return input_path


def test_features_add_sepal_area(tmp_path, sample_df):
    output_path = tmp_path / "output.csv"

    result = engineer_features(
        str(sample_df), str(output_path)
    )

    assert "sepal_area" in result.columns
    assert result.loc[0, "sepal_area"] == pytest.approx(5.1 * 3.5)


def test_features_add_petal_area(tmp_path, sample_df):
    output_path = tmp_path / "output.csv"

    result = engineer_features(
        str(sample_df), str(output_path)
    )

    assert "petal_area" in result.columns
    assert result.loc[0, "petal_area"] == pytest.approx(1.4 * 0.2)


def test_features_add_length_ratio(tmp_path, sample_df):
    output_path = tmp_path / "output.csv"

    result = engineer_features(
        str(sample_df), str(output_path)
    )

    assert "sepal_to_petal_length_ratio" in result.columns
    assert result.loc[0, "sepal_to_petal_length_ratio"] == pytest.approx(
        5.1 / 1.4
    )


def test_features_add_petal_length_bin(tmp_path, sample_df):
    output_path = tmp_path / "output.csv"

    result = engineer_features(
        str(sample_df), str(output_path)
    )

    assert "petal_length_bin" in result.columns
    assert result["petal_length_bin"].notnull().all()
