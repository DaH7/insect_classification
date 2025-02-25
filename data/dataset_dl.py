import os
import zipfile
import kaggle

# Define dataset name and download path
# dataset_name = "hammaadali/insects-recognition"
# download_path = "insects_dataset"
# zip_file = "insects-recognition.zip"

kaggle_api_path =  os.path.expanduser("~/kaggle_key.json")

# download dataset
print("Downloading dataset...")
kaggle.api.dataset_download_files(dataset_name, path=".", unzip=False)

# extract dataset
print("Extracting dataset...")
with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(download_path)

# clean up zip file
os.remove(zip_file)

print(f"Dataset downloaded and extracted to '{download_path}' successfully!")