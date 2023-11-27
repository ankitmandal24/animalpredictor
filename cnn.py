import torch
from torchvision import transforms
from flask import Flask, request, render_template, jsonify
from PIL import Image
import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# from PIL import Image
import glob
from pathlib import Path
from tqdm import tqdm

app = Flask(__name__)


class TinyVGG(nn.Module):
    #Model architecture copying TinyVGG from CNN Explainer

    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size= 3,
                      stride=1,
                      padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size= 3,
                      stride=1,
                      padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride = 2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size= 3,
                      stride=1,
                      padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size= 3,
                      stride=1,
                      padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride = 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13, #mat1 and mat2 shapes cannot be multiplied (32x2560 and 10x2) ot fix this error we multiplide it by 16*16
                      out_features=output_shape)
        )

    def forward(self,x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(X))) #operator fusion

PATH = "entire_model.pt"
model = torch.load(PATH)
model.eval()
class_names = ['cats', 'dogs']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        image = Image.open(file)
        image_tensor = preprocess_image(image)

        with torch.no_grad():
            model.eval()
            predictions = model(image_tensor)
            predicted_class = torch.argmax(predictions).item()
            predicted_class_name = class_names[predicted_class]
            probability = torch.softmax(predictions, dim=1)[0][predicted_class].item()

            response = {
                'class_name': predicted_class_name,
                'probability': probability
            }
            return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
