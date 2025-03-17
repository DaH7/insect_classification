import os
import hashlib
from PIL import Image, ImageFile
import csv
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from skimage import io
import pandas as pd

dataset_path = "C:/Users/dahan/PycharmProjects/insect_classification/data/Insect_classes_dataset_resized"
csv_file = "classification.csv"

def rename_insect_images(main_folder):
    for insect_folder in sorted(os.listdir(main_folder)):
        insect_path = os.path.join(main_folder, insect_folder)

        if os.path.isdir(insect_path):
            files = sorted(os.listdir(insect_path))

            for index, file in enumerate(files, start=1):
                file_path = os.path.join(insect_path, file)

                if os.path.isfile(file_path):  # ensure it's a file
                    new_name = f"{insect_folder}_{index}.jpg"
                    new_path = os.path.join(insect_path, new_name)

                    os.rename(file_path, new_path)  # rename the file
                    print(f"Renamed: {file} → {new_name}")


def get_file_hash(file_path):
    """Returns the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def dupe_check(main_folder):
    hash_map = {}  # stores file hashes and their paths
    for folder in sorted(os.listdir(main_folder)):
        folder_path = os.path.join(main_folder, folder)

        if os.path.isdir(folder_path):
            files = sorted(os.listdir(folder_path))

            for file in sorted(os.listdir(folder_path)):
                if file.lower().endswith((".jpg", ".jpeg")):
                    file_path = os.path.join(folder_path, file)
                    file_hash = get_file_hash(file_path)

                    if file_hash in hash_map:
                        os.remove(file_path)  # delete duplicate file
                        print(f"Deleted duplicate: {file_path}")
                        # print(f"Duplicate found: {file_path} is a duplicate of {hash_map[file_hash]}")
                    else:
                        hash_map[file_hash] = file_path  # stores new hash

def resize_images(main_folder, size=(128, 128)):
    for folder in sorted(os.listdir(main_folder)):
        folder_path = os.path.join(main_folder, folder)

        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if os.path.isfile(filepath):
                try:
                    # opens image
                    img = Image.open(filepath)
                    # checks the image mode and convert to RGB if necessary
                    if img.mode == 'P':  # palettized image (indexed color)
                        img = img.convert('RGB')  # convert to RGB mode
                    # check if the image has an alpha channel (RGBA), convert to RGB
                    elif img.mode == 'RGBA':  # RGBA image
                        img = img.convert('RGB')
                    # resize image
                    img = img.resize(size)
                    # save image
                    img.save(filepath, format='JPEG')
                    print(f"Processed: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")



def folder_classification_csv(main_folder,csv_file):

    folder_names = [folder for folder in sorted(os.listdir(main_folder))
                    if os.path.isdir(os.path.join(main_folder, folder))]
    # creates mapping from folder names to indices
    class_to_idx = {folder: idx for idx, folder in enumerate(folder_names)}

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # header
        writer.writerow(['class_name', 'classification'])

        # write each folder and its corresponding index
        for folder, class_label in class_to_idx.items():
            writer.writerow([folder, class_label])

def image_classification_csv(main_folder, csv_file):
    # list all folders in the main directory
    folder_names = sorted(os.listdir(main_folder))
    class_to_idx = {folder: idx for idx, folder in enumerate(folder_names)}
    # open the CSV file in write mode
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # header
        writer.writerow(['image_name', 'class_name'])
        # iterate through each folder and write to the CSV
        for folder in folder_names:
            folder_path = os.path.join(main_folder, folder)

            # Check if it's a directory
            if os.path.isdir(folder_path):
                # Write the folder name (class_name) and image filenames to the CSV
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    if os.path.isfile(image_path):
                        class_label = class_to_idx[folder]  # Convert class name to number
                        writer.writerow([image_name, class_label])  # Store image name with numeric label






def normalization(main_folder):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    # load dataset
    dataset = datasets.ImageFolder(root=main_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # initialize vars
    mean = torch.zeros(3)
    sum_squared = torch.zeros(3)
    num_samples = 0

    # loops through dataset
    for images, _ in loader:
        batch_samples = images.size(0)  #batch size
        num_samples += batch_samples

        mean += images.mean(dim=[0, 2, 3]) * batch_samples
        sum_squared += (images ** 2).mean(dim=[0, 2, 3]) * batch_samples

    mean /= num_samples
    std = (sum_squared / num_samples - mean ** 2) ** 0.5

    print(f"Mean: {mean.tolist()}")
    print(f"Std: {std.tolist()}")

    return mean, std
"""
before data aug
Mean: [0.563508152961731, 0.5613563656806946, 0.40226054191589355]
Std: [0.2856486439704895, 0.26577502489089966, 0.31352096796035767]
"""






# check_corrupted_images(dataset_path)
# normalization(dataset_path)
folder_classification_csv(dataset_path, csv_file)
# rename_insect_images(dataset_path)
# dupe_check(dataset_path)
# resize_images(dataset_path)!")