import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from pathlib import Path

def download_extract_kaggle(dataset:str, data_path:str, zip_file_name:str):
    """
    Download and extract a dataset from Kaggle using its API.
    Args:
    - dataset (str): Kaggle dataset identifier (e.g., 'username/datasetname').
    - data_path (str): Path to save the dataset.
    - zip_file_name (str): Name for the downloaded zip file.
    """
    image_path = Path(data_path)

    # Create data path if it doesn't exist
    if not image_path.is_dir():
        print(f"Directory {image_path} not found. Creating it...")
        image_path.mkdir(parents=True, exist_ok=True)

    # Authenticate and initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download dataset
    zip_file_path = image_path / zip_file_name
    if not zip_file_path.exists():
        print(f"Downloading {dataset} from Kaggle...")
        api.dataset_download_files(dataset, path=str(image_path), unzip=False)
        print("Download complete.")
    else:
        print(f"{zip_file_name} already exists, skipping download...")

    # Extract the dataset
    print(f"Unzipping {zip_file_name}...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(image_path)
    print("Extraction completed.")
