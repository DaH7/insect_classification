import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from numpy.ma.core import shape
from sympy.physics.units import momentum
from sympy.physics.vector.printing import params
from torch.utils.data import DataLoader
from customDataset import CustomDataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image

def test_image_loading():
    # Hyperparameters
    batch_size = 4  # Load a small batch for testing purposes

    # Define the transforms (now handled by the dataset, but we can still specify them if needed)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize image
        transforms.ToTensor()  # Convert image to tensor
    ])

    # Initialize your dataset
    dataset = CustomDataset(csv_file='C:/Users/dahan/PycharmProjects/insect_classification/data/classification.csv',
                             root_dir='C:/Users/dahan/PycharmProjects/insect_classification/data/Insect_classes_dataset_resized',
                             transform=transform)

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # List to store corrupted image paths
    corrupted_images = []

    # Iterate through the dataset to check for corrupted images
    for i, (images, labels) in enumerate(data_loader):
        for j, (image, label) in enumerate(zip(images, labels)):
            img_path = dataset.image_paths[i * batch_size + j]  # Get the image path from dataset

            try:
                # Attempt to process the image (this will apply the transform)
                # Ensure the image is in RGB format before applying the transform
                img = Image.open(img_path)
                img = img.convert("RGB")  # Convert to RGB if not already in that format

                if transform:
                    img = transform(img)  # Apply the transform

            except (IOError, OSError, ValueError) as e:
                # If an error occurs, log the image path as corrupted
                print(f"Corrupted image detected: {img_path} - Error: {e}")
                corrupted_images.append(img_path)
                continue  # Skip the corrupted image and continue with the next one

    # Report the total number of corrupted images found
    if corrupted_images:
        print(f"Total corrupted images found: {len(corrupted_images)}")
        for img in corrupted_images:
            print(img)
    else:
        print("No corrupted images found.")


def repair_image(img_path):
    try:
        # Open the image
        img = Image.open(img_path)
        img.verify()  # Verify the image integrity (this checks for corruption)

        # Try to save it again to a new file to "rebuild" it
        repaired_path = img_path.replace(".jpg", "_repaired.jpg")
        img = Image.open(img_path)  # Reopen the image after verification
        img.save(repaired_path)
        print(f"Repaired image saved as: {repaired_path}")
    except (IOError, OSError) as e:
        print(f"Failed to repair image: {img_path} - Error: {e}")


if __name__ == "__main__":
    test_image_loading()
    # repair_image('C:/Users/dahan/PycharmProjects/insect_classification/data/Insect_classes_dataset_resized\Fly\Fly_82.jpg')