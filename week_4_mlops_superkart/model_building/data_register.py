from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Define Hugging Face dataset repository
repo_id = "sandeep466/superkart-sales-dataset"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# Upload dataset folder to Hugging Face
api.upload_folder(
    folder_path="week_4_mlops_superkart/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
