import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((128, 128)),  # resize image
            transforms.ToTensor()
        ])
        # load class mappings from CSV
        self.classes = self.csv_to_dict(csv_file)
        # load all image paths and labels
        self.image_paths, self.labels = self.load_images()

    def csv_to_dict(self, csv_file):
        df = pd.read_csv(csv_file)
        return {row['classification']: row['class_name'] for _, row in df.iterrows()}

    def load_images(self):
        image_paths = []
        labels = []
        for _, row in self.annotations.iterrows():
            class_folder = row['class_name']
            class_label = row['classification']
            class_path = os.path.join(self.root_dir, class_folder)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    if os.path.isfile(img_path):
                        image_paths.append(img_path)
                        labels.append(class_label)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = io.imread(img_path)
        image = Image.fromarray(image)  # numpy to PIL
        image = image.convert("RGB")  # force image to RGB

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[index], dtype=torch.long)

        return image, label
