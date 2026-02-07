# for data manipulation
import pandas as pd
import os

# for preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# for model serialization
import joblib

# for Hugging Face Hub interaction
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Initialize Hugging Face API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load training and testing data from Hugging Face
Xtrain_path = "hf://datasets/sandeep466/superkart-sales-dataset/Xtrain.csv"
Xtest_path  = "hf://datasets/sandeep466/superkart-sales-dataset/Xtest.csv"
ytrain_path = "hf://datasets/sandeep466/superkart-sales-dataset/ytrain.csv"
ytest_path  = "hf://datasets/sandeep466/superkart-sales-dataset/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest  = pd.read_csv(ytest_path)

print("Training and testing datasets loaded successfully.")

# Identify numerical and categorical features
numeric_features = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year"
]

categorical_features = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]

# Preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define base model
gbr_model = GradientBoostingRegressor(random_state=42)

# Hyperparameter grid
param_grid = {
    "gradientboostingregressor__n_estimators": [100, 150],
    "gradientboostingregressor__learning_rate": [0.05, 0.1],
    "gradientboostingregressor__max_depth": [3, 4]
}

# Create model pipeline
model_pipeline = make_pipeline(preprocessor, gbr_model)

# Hyperparameter tuning
grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring="neg_root_mean_squared_error"
)

grid_search.fit(Xtrain, ytrain.values.ravel())

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_train_pred = best_model.predict(Xtrain)
y_test_pred = best_model.predict(Xtest)

# Evaluation metrics
train_rmse = mean_squared_error(ytrain, y_train_pred, squared=False)
test_rmse  = mean_squared_error(ytest, y_test_pred, squared=False)
test_mae   = mean_absolute_error(ytest, y_test_pred)
test_r2    = r2_score(ytest, y_test_pred)

print("Model evaluation completed.")
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Test MAE:", test_mae)
print("Test R2:", test_r2)

# Save the best model locally
model_filename = "best_superkart_sales_model_v1.joblib"
joblib.dump(best_model, model_filename)

print(f"Model saved locally as {model_filename}")

# Register model on Hugging Face Model Hub
model_repo_id = "sandeep466/superkart-sales-model"
repo_type = "model"

try:
    api.repo_info(repo_id=model_repo_id, repo_type=repo_type)
    print(f"Model repository '{model_repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Model repository '{model_repo_id}' not found. Creating...")
    create_repo(repo_id=model_repo_id, repo_type=repo_type, private=False)
    print(f"Model repository '{model_repo_id}' created.")

api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id=model_repo_id,
    repo_type=repo_type,
)

print("Model uploaded to Hugging Face Model Hub.")
