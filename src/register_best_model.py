
import mlflow
from mlflow.tracking import MlflowClient

# Connect to the MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Find our experiment
experiment = client.get_experiment_by_name(
    "iris-classification-baseline"
)

if experiment is None:
    raise Exception("Experiment not found!")

# Find the run with the highest accuracy
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1,
)

if not runs:
    raise Exception("No completed runs found!")

best_run = runs[0]
run_id = best_run.info.run_id

print("Best model run:", best_run.data.tags.get("mlflow.runName"))
print("Best accuracy:", best_run.data.metrics["accuracy"])
print("Run ID:", run_id)

# Register the best model
model_uri = f"runs:/{run_id}/model"
model_name = "iris-classifier-prod"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
)

print("Registered model:", registered_model.name)
print("Model version:", registered_model.version)

# Set the model version to Staging
client.transition_model_version_stage(
    name=model_name,
    version=registered_model.version,
    stage="Staging",
)

print("Model successfully moved to Staging!")
