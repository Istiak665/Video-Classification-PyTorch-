import matplotlib
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from train import ImageDataset, testloader, test_data
import torch
import numpy as np
import joblib
import cv2
import cnn_models
import albumentations
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd


# Device check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Computation device: {device}\n")

# load the trained model and label binarizer from disk
print('Loading model and label binarizer...')
lb = joblib.load('../output/lb.pkl')
# print(lb.classes_)
# Import Model
model = cnn_models.CustomCNN().to(device)
# print(model)
# Load model parameters
model.load_state_dict(torch.load('equipments_activities_recognizerv1.pth'))
# Data augmentation
aug = albumentations.Compose([
    albumentations.Resize(224, 224),
    ])

y_true = []
y_pred = []

for data in tqdm(testloader):
    images, labels = data[0].to(device), data[1]
    y_true.extend(labels.numpy())

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)
    y_pred.extend(predicted.cpu().numpy())

cf_matrix = confusion_matrix(y_true, y_pred)
print(cf_matrix)

class_names = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog')
# Create pandas dataframe
dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)

