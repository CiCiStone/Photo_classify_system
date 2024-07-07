import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch import nn
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet152
from dataloader import Data_Loader
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from tqdm import tqdm


def get_raw_model(out_future):
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, out_future)
    return model

def get_trained_model(out_future, pt_path):
    model = get_raw_model(out_future)
    model.load_state_dict(torch.load(pt_path))
    return model

def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # 添加批次维度
    return image

def predict_image(image_path, mynet):
    image = preprocess_image(image_path)
    output = mynet(image)
    _, predicted = torch.max(output.data, 1)
    print("predicted", predicted)
    confidence = torch.softmax(output, dim=1)[0][predicted].item()
    return predicted.item(), confidence

