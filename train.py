# Let’s begin with importing the modules and libraries

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
import albumentations
import torch.optim as optim
import os
import cnn_models
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas as pd
matplotlib.style.use('ggplot')
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# Hyperparameters
lr = 1e-3
batch_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Computation device: {device}\n")
# We are using a learning rate of 0.001 and a batch size of 32

# Read the Data CSV File and Split into Training and Validation Set
# read the data.csv file and get the image paths and labels
df = pd.read_csv('../input/data.csv')
X = df.image_path.values # image paths
y = df.target.values # targets
(xtrain, xtest, ytrain, ytest) = train_test_split(X, y,
    test_size=0.2, random_state=42)
print(f"Training instances: {len(xtrain)}")
print(f"Validation instances: {len(xtest)}")

# Preparing the Custom Dataset
# custom dataset
class ImageDataset(Dataset):
    def __init__(self, images, labels=None, tfms=None):
        self.X = images
        self.y = labels
        # apply augmentations
        if tfms == 0:  # if validating
            self.aug = albumentations.Compose([
                albumentations.Resize(224, 224, always_apply=True),
            ])
        else:  # if training
            self.aug = albumentations.Compose([
                albumentations.Resize(224, 224, always_apply=True),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.3,
                    scale_limit=0.3,
                    rotate_limit=15,
                    p=0.5
                ),
            ])

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        image = Image.open(self.X[i])
        image = image.convert('RGB')
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]
        return (torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long))

# Defining the Training and Validation Data Loaders
train_data = ImageDataset(xtrain, ytrain, tfms=1)
test_data = ImageDataset(xtest, ytest, tfms=0)
# dataloaders
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initializing the Neural Network Model
model = cnn_models.CustomCNN().to(device)
print(model)
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Next, we need to define the loss function and optimizer
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# loss function
criterion = nn.CrossEntropyLoss()

# For better learning, let’s define a learning rate scheduler as well
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        min_lr=1e-6,
        verbose=True
    )

# The Training Function. Here, we will define our training function and call it fit().
# training function
def fit(model, train_dataloader):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in tqdm(enumerate(train_dataloader), total=int(len(train_data) / train_dataloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct / len(train_dataloader.dataset)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")

    return train_loss, train_accuracy

# The Validation Function
# validation function
def validate(model, test_dataloader):
    print('Validating')
    model.eval()
    y_true = []
    y_pred = []
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader), total=int(len(test_data) / test_dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            y_true.extend(target.numpy())
            outputs = model(data)
            loss = criterion(outputs, target)

            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            y_pred.extend(preds.cpu().numpy())

            val_running_correct += (preds == target).sum().item()

        val_loss = val_running_loss / len(test_dataloader.dataset)
        val_accuracy = 100. * val_running_correct / len(test_dataloader.dataset)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')

        return val_loss, val_accuracy, y_true, y_pred


# Executing the Training and Validation Functions for the Specified Number of Epochs
epoch = 10
train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []
start = time.time()
for epoch in range(epoch):
    print(f"Epoch {epoch+1}")
    train_epoch_loss, train_epoch_accuracy = fit(model, trainloader)
    val_epoch_loss, val_epoch_accuracy, y_true, y_pred = validate(model, testloader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    scheduler.step(val_epoch_loss)
end = time.time()
print(f"{(end-start)/60:.3f} minutes")

# Finally, we just need to save the accuracy and loss graphical plots
# accuracy plots
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validataion accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../output/accuracy.png')
plt.show()
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../output/loss.png')
plt.show()

# Confusion Matrix
cf_matrix = confusion_matrix(y_true, y_pred)
print(cf_matrix)

class_names = ('bulldozer_scraping', 'dump_truck_dumping', 'dump_truck_haul', 'excavator_scraping',
               'excavator_truck_loading', 'loader_dumping', 'loader_loading')
# Create pandas dataframe
dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)
# Save Confusion Matrix
# save as CSV file
dataframe.to_csv('../output/confusion_matrix.csv', index=False)

# serialize the model to disk
print('Saving model...')
torch.save(model.state_dict(), "equipments_activities_recognizerv2.pth")

print('TRAINING COMPLETE')
