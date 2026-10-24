
import mlflow
import pandas as pd

# Connect to the MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the registered model from Staging
model_uri = "models:/iris-classifier-prod/1"
model = mlflow.sklearn.load_model(model_uri)

print("Registered model loaded successfully!")

# Prepare sample data using the same seven features used during training
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2, 17.85, 0.28, 7.0]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "sepal_area",
        "petal_area",
        "sepal_to_petal_length_ratio",
    ],
)

# Predict the Iris species
prediction = model.predict(sample)

print("Predicted Iris species:", prediction[0])
