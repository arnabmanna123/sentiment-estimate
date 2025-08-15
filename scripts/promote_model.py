# promote model

import os
import mlflow


def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "vikashdas770"
    repo_name = "sentiment-estimate"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()

    model_name = "my_model"
    # Get the latest version in staging using search_model_versions
    staging_versions = client.search_model_versions(f"name='{model_name}'")
    staging_versions = [
        v for v in staging_versions if v.current_stage == "Staging"]
    if not staging_versions:
        raise Exception("No model version found in Staging")
    latest_version_staging = max(
        staging_versions, key=lambda x: x.creation_timestamp).version

    # Get production versions using search_model_versions
    prod_versions = client.search_model_versions(f"name='{model_name}'")
    prod_versions = [
        v for v in prod_versions if v.current_stage == "Production"]
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")


if __name__ == "__main__":
    promote_model()
