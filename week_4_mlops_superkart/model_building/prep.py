#for data manipulation
import pandas as pd
import os

# for data preprocessing and splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# for Hugging Face Hub interaction
from huggingface_hub import HfApi

# Initialising Hugging Face API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Loading dataset directly from Hugging Face
DATASET_PATH = "hf://datasets/sandeep466/superkart-sales-dataset/SuperKart.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Droping unnecessary identifier columns
df.drop(columns=["Product_Id", "Store_Id"], inplace=True)

# Encoding categoricaal features
categorical_cols = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

#Defining target variable
target_col = "Product_Store_Sales_Total"

#Splitting into features and target
X = df.drop(columns=[target_col])
y = df[target_col]

#Performing train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Saving prepared datasets locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

#Uploading prepared datasets back to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="sandeep466/superkart-sales-dataset",
        repo_type="dataset",
    )
