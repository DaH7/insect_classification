import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from numpy.ma.core import shape
from torch.utils.data import DataLoader
from customDataset import insect_dataset
import matplotlib.pyplot as plt

#setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#hyperparameters
in_channels= 3
num_classes = 10
learning_rate = 1e-3
batch_size = 32
num_epochs = 1

#load data
dataset = insect_dataset(csv_file='all_insects.csv',
                         root_dir='all_insects',
                         transform = transforms.ToTensor())

train_set,test_set = torch.utils.data.random_split(dataset, [3300,514])
train_loader = DataLoader(dataset = train_set,batch_size=batch_size,shuffle = True)
test_loader = DataLoader(dataset = train_set,batch_size=batch_size,shuffle = True)


# Get the first batch from train_loader
first_batch = next(iter(train_loader))
images, labels = first_batch
image, label = images[0], labels[0]
class_names = dataset.classes
#making sure  everything works
#print(image.shape) #(3,128,128) color_channels,height,width

# plt.imshow(image.permute(1, 2, 0))  # Rearranges channels for matplotlib (H,W,C)
# plt.title(f"Label: {label.item()}")
# plt.show() #displays correctly

# label = torch.tensor(label, dtype=torch.long)  # Ensure integer class labels
#
# for images, labels in train_loader:
#     print(images.shape)  # expected: (batch_size, C, H, W)
#     print(labels.shape)  # expected: (batch_size,)
#     break

train_feature_batch ,train_labels_batch = next(iter(train_loader))
print(train_feature_batch.shape , train_labels_batch.shape)


#flatten layer (3,16384)
flatten_model = nn.Flatten()
x = train_feature_batch[0]
output = flatten_model(x)
# print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
# print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")
# print(x)
# print(output)

class insectModelV0(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_shape)
        )
    def forward(self,x):
        return self.layer_stack(x)

#small test run
torch.manual_seed(42)

model_0 = insectModelV0(input_shape=16384, # one for every pixel (28x28)
    hidden_units=10, # how many units in the hidden layer
    output_shape=len(class_names) # one for every class
)

