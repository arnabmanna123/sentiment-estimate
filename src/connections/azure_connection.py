import pandas as pd
import logging
from src.logger import logging
from io import StringIO
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError


class azure_operations:
    def __init__(self, connection_string, container_name):
        """
        Initialize the azure_operations class with Azure Storage credentials and container details.
        """
        self.container_name = container_name
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                connection_string)
            self.container_client = self.blob_service_client.get_container_client(
                container_name)
            logging.info("Data Ingestion from Azure Blob Storage initialized")
        except Exception as e:
            logging.exception(
                f"Failed to initialize Azure Blob Storage connection: {e}")
            raise

    def fetch_file_from_azure(self, file_path):
        """
        Fetches a CSV file from Azure Blob Storage and returns it as a Pandas DataFrame.
        :param file_path: Blob file path (e.g., 'data/data.csv')
        :return: Pandas DataFrame
        """
        try:
            logging.info(
                f"Fetching file '{file_path}' from container '{self.container_name}'...")
            blob_client = self.container_client.get_blob_client(file_path)

            # Download the blob content
            download_stream = blob_client.download_blob()
            file_content = download_stream.readall().decode('utf-8')

            # Convert to DataFrame
            df = pd.read_csv(StringIO(file_content))
            logging.info(
                f"Successfully fetched and loaded '{file_path}' from Azure that has {len(df)} records.")
            return df

        except AzureError as ae:
            logging.exception(
                f"❌ Azure-specific error while fetching '{file_path}': {ae}")
            return None
        except Exception as e:
            logging.exception(
                f"❌ Failed to fetch '{file_path}' from Azure: {e}")
            return None

    def upload_file_to_azure(self, file_path, local_file_path):
        """
        Uploads a file to Azure Blob Storage.
        :param file_path: Destination path in the blob container
        :param local_file_path: Path to the local file to upload
        :return: bool indicating success/failure
        """
        try:
            logging.info(
                f"Uploading file to '{file_path}' in container '{self.container_name}'...")
            blob_client = self.container_client.get_blob_client(file_path)

            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            logging.info(f"Successfully uploaded file to '{file_path}'")
            return True

        except AzureError as ae:
            logging.exception(
                f"❌ Azure-specific error while uploading to '{file_path}': {ae}")
            return False
        except Exception as e:
            logging.exception(f"❌ Failed to upload to '{file_path}': {e}")
            return False

# Example usage
# if __name__ == "__main__":
#     # Replace these with your actual Azure Storage credentials and details
#     CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
#     CONTAINER_NAME = "your-container-name"
#     FILE_PATH = "data/data.csv"  # Path inside blob container
#
#     azure_client = azure_operations(CONNECTION_STRING, CONTAINER_NAME)
#
#     # Fetch data
#     df = azure_client.fetch_file_from_azure(FILE_PATH)
#     if df is not None:
#         print(f"Data fetched with {len(df)} records.")
#
#     # Upload example
#     # success = azure_client.upload_file_to_azure("data/new_file.csv", "local_path/file.csv")
#     # if success:
#     #     print("File uploaded successfully")
