import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from numpy.ma.core import shape
from sympy.physics.units import momentum
from sympy.physics.vector.printing import params
from torch.utils.data import DataLoader
from customDataset import insect_dataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

#setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#hyperparameters
lr = 1e-3
momentum =.9
batch_size = 32


#load data
dataset = insect_dataset(csv_file='all_insects.csv',
                         root_dir='all_insects',
                         transform = transforms.ToTensor())

train_set,test_set = torch.utils.data.random_split(dataset, [3300,514])
train_loader = DataLoader(dataset = train_set,
                          batch_size=batch_size,
                          shuffle = True)

test_loader = DataLoader(dataset = train_set,
                         batch_size=batch_size,
                         shuffle = True)



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

#intial model
model_0 = insectModelV0(input_shape=49152, # one for every pixel (128*128)
    hidden_units=10, # how many units in the hidden layer
    output_shape=len(class_names) # one for every class
)

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


#loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=lr, momentum=momentum)
# optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

#small test run
torch.manual_seed(42)
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-----")

    #training
    train_loss = 0

    for train_feature_batch, (X,y) in enumerate(train_loader):
        model_0.train()
        #forward pass
        y_pred = model_0(X)
        #Calculate loss per batch
        loss = loss_fn(y_pred,y)
        train_loss += loss #aggregates loss per epoch
        #optimizer zero gradiant
        optimizer.zero_grad()
        #backward loss
        loss.backward()
        #optimizer step
        optimizer.step()
        # how many samples have been gone through
        if train_feature_batch % 400 == 0:
            print(f"Looked at {train_feature_batch * len(X)}/{len(train_loader.dataset)} samples")

        #avg loss per batch per epoch
        train_loss /= len(train_loader)

        #testing
        test_loss, test_acc = 0,0
        model_0.eval()

        with torch.inference_mode():
            for X,y in test_loader:
                #forward pass
                test_pred = model_0(X)
                #calculate loss
                test_loss += loss_fn(test_pred,y)
                #calculate accuracy
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
            #calulate test metrics
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
        #print results
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")





