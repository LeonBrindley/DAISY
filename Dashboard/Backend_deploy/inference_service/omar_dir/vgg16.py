import torch
from torchvision.models import vgg16
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
# import keras
import os
import inference_service.omar_dir.dropout as dropout

# import inference_service.omar_dir.fine_tune

class PerceptronMultiLabelVGG16(torch.nn.Module):
    def __init__(self, original_model, num_classes):
        super(PerceptronMultiLabelVGG16, self).__init__()
        self.vgg16 = original_model
        self.vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=512)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.vgg16(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class MultiLabelVGG16(nn.Module):
    def __init__(self, original_model, num_classes):
        super(MultiLabelVGG16, self).__init__()
        self.vgg16 = original_model
        self.vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.vgg16(x)
        x = self.sigmoid(x)
        return x

def load_VGG16_model():
  return vgg16(pretrained=True)

def load(use_extra_perceptron, dropout_p):
    model = load_VGG16_model()

    new_num_classes = 4  # {Clover, Grass, Dung, Soil}

    model = PerceptronMultiLabelVGG16(model, new_num_classes) if use_extra_perceptron else MultiLabelVGG16(model, new_num_classes)

    return model

# experiment params:
# unfreeze_layers: classifier.6, classifier.3, classifier.0, features.28, features.26, features.24, features.21
# use_extra_perceptron: True, False
# learning_rate: 0.001, 0.0001, 0.00001, 0.000001, 0.0000001
# epochs: 30
# patience: 5
def fine_tune_vgg16(
    train_loader, val_loader, use_extra_perceptron, unfreeze_layers,
    learning_rate=0.001, epochs=10, patience=5):

    model = load(use_extra_perceptron)

    # freeze all weights
    fine_tune.freeze_weights(model)

    # unfreeze final layers for finetuning
    fine_tune.unfreeze_weights(model, unfreeze_layers)

    # Define loss function and optimiser
    loss_function = nn.BCELoss()  # Binary cross-entropy loss for one-hot labels
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)  # Adjust learning rate as needed

    # Detect if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Fine-tune the model
    model, train_stats, val_stats = fine_tune.fine_tune_model(
        model,
        device,
        train_loader, 
        val_loader, 
        optimiser, 
        loss_function, 
        num_epochs=epochs,
        patience=patience)


    return model, train_stats, val_stats


# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )
# PARAMS
# features.0.weight
# features.0.bias
# features.2.weight
# features.2.bias
# features.5.weight
# features.5.bias
# features.7.weight
# features.7.bias
# features.10.weight
# features.10.bias
# features.12.weight
# features.12.bias
# features.14.weight
# features.14.bias
# features.17.weight
# features.17.bias
# features.19.weight
# features.19.bias
# features.21.weight
# features.21.bias
# features.24.weight
# features.24.bias
# features.26.weight
# features.26.bias
# features.28.weight
# features.28.bias
# classifier.0.weight
# classifier.0.bias
# classifier.3.weight
# classifier.3.bias
# classifier.6.weight
# classifier.6.bias