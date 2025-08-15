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

    try:
        # Get all versions of the model
        versions = client.get_registered_model(model_name).versions
        if not versions:
            raise Exception(f"No versions found for model {model_name}")

        # Get the latest version
        latest_version = max(versions, key=lambda x: x.creation_timestamp)
        version_number = latest_version.version

        # Set the 'production' alias for the latest version
        client.set_registered_model_alias(
            model_name, "production", version_number)

        print(f"Model version {version_number} set as production alias")
    except Exception as e:
        print(f"Error promoting model: {str(e)}")
        raise


if __name__ == "__main__":
    promote_model()
