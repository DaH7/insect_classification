import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import torchvision.transforms as transforms

class insect_dataset(Dataset):
    def __init__(self,csv_file,root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {0:"Butterfly", 1:"Dragonfly", 2:"Grasshopper",3:"Ladybug",4:"Mosquito"}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        else:
            image=transforms.ToTensor()(image)

        label = torch.tensor(int(self.annotations.iloc[index,1]),dtype=torch.long)

        return(image,y_label)