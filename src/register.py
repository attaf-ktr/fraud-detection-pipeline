import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

best = mlflow.search_runs(
    experiment_names=["fraud-detection"],
    order_by=["metrics.pr_auc DESC"],
    max_results=1,
).iloc[0]

print("Best run:", best["tags.mlflow.runName"], "pr_auc =", round(best["metrics.pr_auc"], 4))

mv = mlflow.register_model(f"runs:/{best['run_id']}/model", "fraud-scoring")
client.set_registered_model_alias("fraud-scoring", "champion", mv.version)
print("Registered fraud-scoring version", mv.version, "as @champion")
