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
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


# allow multiprocessing
torch.multiprocessing.freeze_support()

def main():
    #setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    #hyperparameters
    lr = 0.001
    momentum = 0.9
    batch_size = 32
    epochs = 30

    #load data
    dataset = insect_dataset(csv_file='all_insects.csv',
                             root_dir='all_insects',
                             transform = transforms.ToTensor())

    train_set,test_set = torch.utils.data.random_split(dataset, [3300,514])
    train_loader = DataLoader(dataset = train_set,
                              batch_size=batch_size,
                              shuffle = True,
                              num_workers=4,  # Increase based on available CPU cores
                              pin_memory=True  # Speeds up GPU data transfer
                              )

    test_loader = DataLoader(dataset = test_set,
                             batch_size=batch_size,
                             shuffle = False,
                             num_workers=4,
                             pin_memory=True
                             )



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
            self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels = input_shape,
                          out_channels= hidden_units,
                          kernel_size = 3 ,
                          stride = 1,
                          padding = 1),

                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2,
                             stride = 2),

            )
            self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels = hidden_units,
                          out_channels = hidden_units,
                          kernel_size = 3,
                          padding = 1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          padding=1),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.MaxPool2d(kernel_size = 2,
                             stride = 2)
            )

            # Find flattened output size dynamically
            self._compute_flattened_size(input_shape)

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features = self._to_linear,
                          out_features = output_shape)
            )

        def _compute_flattened_size(self, input_channels):
            with torch.no_grad():
                sample_input = torch.randn(1, input_channels, 128, 128)  # Batch size 1, image size 128x128
                out = self.block_1(sample_input)
                out = self.block_2(out)
                self._to_linear = out.view(1, -1).shape[1]  # Flatten and get final feature size

        def forward(self,x:torch.Tensor):
            x = self.block_1(x)
            x = self.block_2(x)
            x = self.classifier(x)
            return x



    # model
    model_0 = insectModelV0(
        input_shape=3, #RGB
        hidden_units=32, # how many units in the hidden layer
        output_shape=len(class_names) # one for every class
        ).to(device)


    # Calculate accuracy (a classification metric)
    def accuracy_fn(y_true, y_pred):

        correct = (y_true == y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

        #correct = torch.eq(y_true, y_pred).sum().item()  #Sum of correct predictions
        # acc = (correct / len(y_pred)) * 100 # Divide by total predictions and convert to percentage
        # return acc


    #loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=lr, momentum=momentum)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=7, factor=0.1)
    # scheduler = StepLR(optimizer, step_size = 15, gamma = 0.1)



    #small test run
    torch.manual_seed(42)
    best_test_loss = float('inf')
    best_model_state = None


    for epoch in tqdm(range(epochs)):
        print(f"\nEpoch: {epoch}\n-----")

        #training
        train_loss = 0

        for batch, (X,y) in enumerate(train_loader):
            model_0.train()

            X, y = X.to(device), y.to(device)

            #forward pass
            y_pred = model_0(X)

            #Calculate loss per batch
            loss = loss_fn(y_pred,y)
            train_loss += loss.item() #aggregates loss per epoch

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            # how many samples have been gone through
            # if batch % 400 == 0:
            #     print(f"Looked at {batch * len(X)}/{len(train_loader.dataset)} samples")



        #avg loss per batch per epoch
        train_loss /= len(train_loader)


        #eval mode
        model_0.eval()

        #initilze
        test_loss, test_acc = 0,0


        with torch.inference_mode():
            for X,y in test_loader:

                X,y = X.to(device), y.to(device)


                #forward pass
                test_pred = model_0(X)

                #calculate loss
                test_loss += loss_fn(test_pred,y).item()


                #calculate accuracy
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

            #calulate test metrics
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)

            val_loss = test_loss
            scheduler.step(val_loss)  # Update learning rate
            print(f"\nLearning rate scheduler step executed. Validation loss: {val_loss:.5f}")

        #print results
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model_0.state_dict()  # Save the model state
            torch.save(best_model_state, 'best_model.pth')  # Save the model to file
            print(f"\nBest model saved with Test Loss: {test_loss:.5f}")

if __name__ == '__main__':
    main()




