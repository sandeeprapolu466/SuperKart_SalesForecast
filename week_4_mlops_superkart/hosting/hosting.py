from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="week_4_mlops_superkart/deployment",
    repo_id="sandeep466/SuperKart-SalesForecast",
    repo_type="space",
    path_in_repo="",
)
