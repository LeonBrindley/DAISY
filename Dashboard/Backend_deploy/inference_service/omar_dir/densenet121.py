import torch
from torchvision.models import densenet121
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
# import keras
import os
import inference_service.omar_dir.dropout as dropout

class PerceptronMultiLabelDenseNet(torch.nn.Module):
    def __init__(self, original_model, num_classes):
        super(PerceptronMultiLabelDenseNet, self).__init__()
        self.densenet = original_model
        self.densenet.classifier = nn.Linear(in_features=self.densenet.classifier.in_features, out_features=512)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class MultiLabelDenseNet(nn.Module):
    def __init__(self, original_model, num_classes):
        super(MultiLabelDenseNet, self).__init__()
        self.densenet = original_model
        self.densenet.classifier = nn.Linear(in_features=self.densenet.classifier.in_features, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet(x)
        x = self.sigmoid(x)
        return x

def load_densenet121_model():
  return densenet121(pretrained=True)

def load(use_extra_perceptron, dropout_p):
    model = load_densenet121_model()

    new_num_classes = 4  # {Clover, Grass, Dung, Soil}

    model = PerceptronMultiLabelDenseNet(model, new_num_classes) if use_extra_perceptron else MultiLabelDenseNet(model, new_num_classes)

    return model

# experiment params:
# unfreeze_layers: classifier, features.norm5, features.denseblock4
# use_extra_perceptron: True, False
# learning_rate: 0.001, 0.0001, 0.00001, 0.000001, 0.0000001
# epochs: 30
# patience: 5


# DenseNet(
#   (features): Sequential(
#     (conv0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     (norm0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu0): ReLU(inplace=True)
#     (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#     (denseblock1): _DenseBlock(
#       (denselayer1): _DenseLayer(
#         (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer2): _DenseLayer(
#         (norm1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(96, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer3): _DenseLayer(
#         (norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer4): _DenseLayer(
#         (norm1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(160, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer5): _DenseLayer(
#         (norm1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer6): _DenseLayer(
#         (norm1): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(224, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#     )
#     (transition1): _Transition(
#       (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
#     )
#     (denseblock2): _DenseBlock(
#       (denselayer1): _DenseLayer(
#         (norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer2): _DenseLayer(
#         (norm1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(160, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer3): _DenseLayer(
#         (norm1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer4): _DenseLayer(
#         (norm1): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(224, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer5): _DenseLayer(
#         (norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer6): _DenseLayer(
#         (norm1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(288, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer7): _DenseLayer(
#         (norm1): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer8): _DenseLayer(
#         (norm1): BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(352, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer9): _DenseLayer(
#         (norm1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer10): _DenseLayer(
#         (norm1): BatchNorm2d(416, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(416, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer11): _DenseLayer(
#         (norm1): BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(448, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer12): _DenseLayer(
#         (norm1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(480, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#     )
#     (transition2): _Transition(
#       (norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
#     )
#     (denseblock3): _DenseBlock(
#       (denselayer1): _DenseLayer(
#         (norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer2): _DenseLayer(
#         (norm1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(288, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer3): _DenseLayer(
#         (norm1): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer4): _DenseLayer(
#         (norm1): BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(352, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer5): _DenseLayer(
#         (norm1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer6): _DenseLayer(
#         (norm1): BatchNorm2d(416, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(416, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer7): _DenseLayer(
#         (norm1): BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(448, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer8): _DenseLayer(
#         (norm1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(480, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer9): _DenseLayer(
#         (norm1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer10): _DenseLayer(
#         (norm1): BatchNorm2d(544, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(544, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer11): _DenseLayer(
#         (norm1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer12): _DenseLayer(
#         (norm1): BatchNorm2d(608, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer13): _DenseLayer(
#         (norm1): BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(640, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer14): _DenseLayer(
#         (norm1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(672, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer15): _DenseLayer(
#         (norm1): BatchNorm2d(704, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(704, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer16): _DenseLayer(
#         (norm1): BatchNorm2d(736, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(736, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer17): _DenseLayer(
#         (norm1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer18): _DenseLayer(
#         (norm1): BatchNorm2d(800, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(800, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer19): _DenseLayer(
#         (norm1): BatchNorm2d(832, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer20): _DenseLayer(
#         (norm1): BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(864, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer21): _DenseLayer(
#         (norm1): BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(896, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer22): _DenseLayer(
#         (norm1): BatchNorm2d(928, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(928, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer23): _DenseLayer(
#         (norm1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer24): _DenseLayer(
#         (norm1): BatchNorm2d(992, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(992, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#     )
#     (transition3): _Transition(
#       (norm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
#     )
#     (denseblock4): _DenseBlock(
#       (denselayer1): _DenseLayer(
#         (norm1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer2): _DenseLayer(
#         (norm1): BatchNorm2d(544, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(544, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer3): _DenseLayer(
#         (norm1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer4): _DenseLayer(
#         (norm1): BatchNorm2d(608, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer5): _DenseLayer(
#         (norm1): BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(640, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer6): _DenseLayer(
#         (norm1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(672, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer7): _DenseLayer(
#         (norm1): BatchNorm2d(704, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(704, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer8): _DenseLayer(
#         (norm1): BatchNorm2d(736, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(736, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer9): _DenseLayer(
#         (norm1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer10): _DenseLayer(
#         (norm1): BatchNorm2d(800, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(800, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer11): _DenseLayer(
#         (norm1): BatchNorm2d(832, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer12): _DenseLayer(
#         (norm1): BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(864, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer13): _DenseLayer(
#         (norm1): BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(896, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer14): _DenseLayer(
#         (norm1): BatchNorm2d(928, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(928, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer15): _DenseLayer(
#         (norm1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#       (denselayer16): _DenseLayer(
#         (norm1): BatchNorm2d(992, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(992, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       )
#     )
#     (norm5): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   )
#   (classifier): Linear(in_features=1024, out_features=1000, bias=True)
# )
# PARAMS
# features.conv0.weight
# features.norm0.weight
# features.norm0.bias
# features.denseblock1.denselayer1.norm1.weight
# features.denseblock1.denselayer1.norm1.bias
# features.denseblock1.denselayer1.conv1.weight
# features.denseblock1.denselayer1.norm2.weight
# features.denseblock1.denselayer1.norm2.bias
# features.denseblock1.denselayer1.conv2.weight
# features.denseblock1.denselayer2.norm1.weight
# features.denseblock1.denselayer2.norm1.bias
# features.denseblock1.denselayer2.conv1.weight
# features.denseblock1.denselayer2.norm2.weight
# features.denseblock1.denselayer2.norm2.bias
# features.denseblock1.denselayer2.conv2.weight
# features.denseblock1.denselayer3.norm1.weight
# features.denseblock1.denselayer3.norm1.bias
# features.denseblock1.denselayer3.conv1.weight
# features.denseblock1.denselayer3.norm2.weight
# features.denseblock1.denselayer3.norm2.bias
# features.denseblock1.denselayer3.conv2.weight
# features.denseblock1.denselayer4.norm1.weight
# features.denseblock1.denselayer4.norm1.bias
# features.denseblock1.denselayer4.conv1.weight
# features.denseblock1.denselayer4.norm2.weight
# features.denseblock1.denselayer4.norm2.bias
# features.denseblock1.denselayer4.conv2.weight
# features.denseblock1.denselayer5.norm1.weight
# features.denseblock1.denselayer5.norm1.bias
# features.denseblock1.denselayer5.conv1.weight
# features.denseblock1.denselayer5.norm2.weight
# features.denseblock1.denselayer5.norm2.bias
# features.denseblock1.denselayer5.conv2.weight
# features.denseblock1.denselayer6.norm1.weight
# features.denseblock1.denselayer6.norm1.bias
# features.denseblock1.denselayer6.conv1.weight
# features.denseblock1.denselayer6.norm2.weight
# features.denseblock1.denselayer6.norm2.bias
# features.denseblock1.denselayer6.conv2.weight
# features.transition1.norm.weight
# features.transition1.norm.bias
# features.transition1.conv.weight
# features.denseblock2.denselayer1.norm1.weight
# features.denseblock2.denselayer1.norm1.bias
# features.denseblock2.denselayer1.conv1.weight
# features.denseblock2.denselayer1.norm2.weight
# features.denseblock2.denselayer1.norm2.bias
# features.denseblock2.denselayer1.conv2.weight
# features.denseblock2.denselayer2.norm1.weight
# features.denseblock2.denselayer2.norm1.bias
# features.denseblock2.denselayer2.conv1.weight
# features.denseblock2.denselayer2.norm2.weight
# features.denseblock2.denselayer2.norm2.bias
# features.denseblock2.denselayer2.conv2.weight
# features.denseblock2.denselayer3.norm1.weight
# features.denseblock2.denselayer3.norm1.bias
# features.denseblock2.denselayer3.conv1.weight
# features.denseblock2.denselayer3.norm2.weight
# features.denseblock2.denselayer3.norm2.bias
# features.denseblock2.denselayer3.conv2.weight
# features.denseblock2.denselayer4.norm1.weight
# features.denseblock2.denselayer4.norm1.bias
# features.denseblock2.denselayer4.conv1.weight
# features.denseblock2.denselayer4.norm2.weight
# features.denseblock2.denselayer4.norm2.bias
# features.denseblock2.denselayer4.conv2.weight
# features.denseblock2.denselayer5.norm1.weight
# features.denseblock2.denselayer5.norm1.bias
# features.denseblock2.denselayer5.conv1.weight
# features.denseblock2.denselayer5.norm2.weight
# features.denseblock2.denselayer5.norm2.bias
# features.denseblock2.denselayer5.conv2.weight
# features.denseblock2.denselayer6.norm1.weight
# features.denseblock2.denselayer6.norm1.bias
# features.denseblock2.denselayer6.conv1.weight
# features.denseblock2.denselayer6.norm2.weight
# features.denseblock2.denselayer6.norm2.bias
# features.denseblock2.denselayer6.conv2.weight
# features.denseblock2.denselayer7.norm1.weight
# features.denseblock2.denselayer7.norm1.bias
# features.denseblock2.denselayer7.conv1.weight
# features.denseblock2.denselayer7.norm2.weight
# features.denseblock2.denselayer7.norm2.bias
# features.denseblock2.denselayer7.conv2.weight
# features.denseblock2.denselayer8.norm1.weight
# features.denseblock2.denselayer8.norm1.bias
# features.denseblock2.denselayer8.conv1.weight
# features.denseblock2.denselayer8.norm2.weight
# features.denseblock2.denselayer8.norm2.bias
# features.denseblock2.denselayer8.conv2.weight
# features.denseblock2.denselayer9.norm1.weight
# features.denseblock2.denselayer9.norm1.bias
# features.denseblock2.denselayer9.conv1.weight
# features.denseblock2.denselayer9.norm2.weight
# features.denseblock2.denselayer9.norm2.bias
# features.denseblock2.denselayer9.conv2.weight
# features.denseblock2.denselayer10.norm1.weight
# features.denseblock2.denselayer10.norm1.bias
# features.denseblock2.denselayer10.conv1.weight
# features.denseblock2.denselayer10.norm2.weight
# features.denseblock2.denselayer10.norm2.bias
# features.denseblock2.denselayer10.conv2.weight
# features.denseblock2.denselayer11.norm1.weight
# features.denseblock2.denselayer11.norm1.bias
# features.denseblock2.denselayer11.conv1.weight
# features.denseblock2.denselayer11.norm2.weight
# features.denseblock2.denselayer11.norm2.bias
# features.denseblock2.denselayer11.conv2.weight
# features.denseblock2.denselayer12.norm1.weight
# features.denseblock2.denselayer12.norm1.bias
# features.denseblock2.denselayer12.conv1.weight
# features.denseblock2.denselayer12.norm2.weight
# features.denseblock2.denselayer12.norm2.bias
# features.denseblock2.denselayer12.conv2.weight
# features.transition2.norm.weight
# features.transition2.norm.bias
# features.transition2.conv.weight
# features.denseblock3.denselayer1.norm1.weight
# features.denseblock3.denselayer1.norm1.bias
# features.denseblock3.denselayer1.conv1.weight
# features.denseblock3.denselayer1.norm2.weight
# features.denseblock3.denselayer1.norm2.bias
# features.denseblock3.denselayer1.conv2.weight
# features.denseblock3.denselayer2.norm1.weight
# features.denseblock3.denselayer2.norm1.bias
# features.denseblock3.denselayer2.conv1.weight
# features.denseblock3.denselayer2.norm2.weight
# features.denseblock3.denselayer2.norm2.bias
# features.denseblock3.denselayer2.conv2.weight
# features.denseblock3.denselayer3.norm1.weight
# features.denseblock3.denselayer3.norm1.bias
# features.denseblock3.denselayer3.conv1.weight
# features.denseblock3.denselayer3.norm2.weight
# features.denseblock3.denselayer3.norm2.bias
# features.denseblock3.denselayer3.conv2.weight
# features.denseblock3.denselayer4.norm1.weight
# features.denseblock3.denselayer4.norm1.bias
# features.denseblock3.denselayer4.conv1.weight
# features.denseblock3.denselayer4.norm2.weight
# features.denseblock3.denselayer4.norm2.bias
# features.denseblock3.denselayer4.conv2.weight
# features.denseblock3.denselayer5.norm1.weight
# features.denseblock3.denselayer5.norm1.bias
# features.denseblock3.denselayer5.conv1.weight
# features.denseblock3.denselayer5.norm2.weight
# features.denseblock3.denselayer5.norm2.bias
# features.denseblock3.denselayer5.conv2.weight
# features.denseblock3.denselayer6.norm1.weight
# features.denseblock3.denselayer6.norm1.bias
# features.denseblock3.denselayer6.conv1.weight
# features.denseblock3.denselayer6.norm2.weight
# features.denseblock3.denselayer6.norm2.bias
# features.denseblock3.denselayer6.conv2.weight
# features.denseblock3.denselayer7.norm1.weight
# features.denseblock3.denselayer7.norm1.bias
# features.denseblock3.denselayer7.conv1.weight
# features.denseblock3.denselayer7.norm2.weight
# features.denseblock3.denselayer7.norm2.bias
# features.denseblock3.denselayer7.conv2.weight
# features.denseblock3.denselayer8.norm1.weight
# features.denseblock3.denselayer8.norm1.bias
# features.denseblock3.denselayer8.conv1.weight
# features.denseblock3.denselayer8.norm2.weight
# features.denseblock3.denselayer8.norm2.bias
# features.denseblock3.denselayer8.conv2.weight
# features.denseblock3.denselayer9.norm1.weight
# features.denseblock3.denselayer9.norm1.bias
# features.denseblock3.denselayer9.conv1.weight
# features.denseblock3.denselayer9.norm2.weight
# features.denseblock3.denselayer9.norm2.bias
# features.denseblock3.denselayer9.conv2.weight
# features.denseblock3.denselayer10.norm1.weight
# features.denseblock3.denselayer10.norm1.bias
# features.denseblock3.denselayer10.conv1.weight
# features.denseblock3.denselayer10.norm2.weight
# features.denseblock3.denselayer10.norm2.bias
# features.denseblock3.denselayer10.conv2.weight
# features.denseblock3.denselayer11.norm1.weight
# features.denseblock3.denselayer11.norm1.bias
# features.denseblock3.denselayer11.conv1.weight
# features.denseblock3.denselayer11.norm2.weight
# features.denseblock3.denselayer11.norm2.bias
# features.denseblock3.denselayer11.conv2.weight
# features.denseblock3.denselayer12.norm1.weight
# features.denseblock3.denselayer12.norm1.bias
# features.denseblock3.denselayer12.conv1.weight
# features.denseblock3.denselayer12.norm2.weight
# features.denseblock3.denselayer12.norm2.bias
# features.denseblock3.denselayer12.conv2.weight
# features.denseblock3.denselayer13.norm1.weight
# features.denseblock3.denselayer13.norm1.bias
# features.denseblock3.denselayer13.conv1.weight
# features.denseblock3.denselayer13.norm2.weight
# features.denseblock3.denselayer13.norm2.bias
# features.denseblock3.denselayer13.conv2.weight
# features.denseblock3.denselayer14.norm1.weight
# features.denseblock3.denselayer14.norm1.bias
# features.denseblock3.denselayer14.conv1.weight
# features.denseblock3.denselayer14.norm2.weight
# features.denseblock3.denselayer14.norm2.bias
# features.denseblock3.denselayer14.conv2.weight
# features.denseblock3.denselayer15.norm1.weight
# features.denseblock3.denselayer15.norm1.bias
# features.denseblock3.denselayer15.conv1.weight
# features.denseblock3.denselayer15.norm2.weight
# features.denseblock3.denselayer15.norm2.bias
# features.denseblock3.denselayer15.conv2.weight
# features.denseblock3.denselayer16.norm1.weight
# features.denseblock3.denselayer16.norm1.bias
# features.denseblock3.denselayer16.conv1.weight
# features.denseblock3.denselayer16.norm2.weight
# features.denseblock3.denselayer16.norm2.bias
# features.denseblock3.denselayer16.conv2.weight
# features.denseblock3.denselayer17.norm1.weight
# features.denseblock3.denselayer17.norm1.bias
# features.denseblock3.denselayer17.conv1.weight
# features.denseblock3.denselayer17.norm2.weight
# features.denseblock3.denselayer17.norm2.bias
# features.denseblock3.denselayer17.conv2.weight
# features.denseblock3.denselayer18.norm1.weight
# features.denseblock3.denselayer18.norm1.bias
# features.denseblock3.denselayer18.conv1.weight
# features.denseblock3.denselayer18.norm2.weight
# features.denseblock3.denselayer18.norm2.bias
# features.denseblock3.denselayer18.conv2.weight
# features.denseblock3.denselayer19.norm1.weight
# features.denseblock3.denselayer19.norm1.bias
# features.denseblock3.denselayer19.conv1.weight
# features.denseblock3.denselayer19.norm2.weight
# features.denseblock3.denselayer19.norm2.bias
# features.denseblock3.denselayer19.conv2.weight
# features.denseblock3.denselayer20.norm1.weight
# features.denseblock3.denselayer20.norm1.bias
# features.denseblock3.denselayer20.conv1.weight
# features.denseblock3.denselayer20.norm2.weight
# features.denseblock3.denselayer20.norm2.bias
# features.denseblock3.denselayer20.conv2.weight
# features.denseblock3.denselayer21.norm1.weight
# features.denseblock3.denselayer21.norm1.bias
# features.denseblock3.denselayer21.conv1.weight
# features.denseblock3.denselayer21.norm2.weight
# features.denseblock3.denselayer21.norm2.bias
# features.denseblock3.denselayer21.conv2.weight
# features.denseblock3.denselayer22.norm1.weight
# features.denseblock3.denselayer22.norm1.bias
# features.denseblock3.denselayer22.conv1.weight
# features.denseblock3.denselayer22.norm2.weight
# features.denseblock3.denselayer22.norm2.bias
# features.denseblock3.denselayer22.conv2.weight
# features.denseblock3.denselayer23.norm1.weight
# features.denseblock3.denselayer23.norm1.bias
# features.denseblock3.denselayer23.conv1.weight
# features.denseblock3.denselayer23.norm2.weight
# features.denseblock3.denselayer23.norm2.bias
# features.denseblock3.denselayer23.conv2.weight
# features.denseblock3.denselayer24.norm1.weight
# features.denseblock3.denselayer24.norm1.bias
# features.denseblock3.denselayer24.conv1.weight
# features.denseblock3.denselayer24.norm2.weight
# features.denseblock3.denselayer24.norm2.bias
# features.denseblock3.denselayer24.conv2.weight
# features.transition3.norm.weight
# features.transition3.norm.bias
# features.transition3.conv.weight
# features.denseblock4.denselayer1.norm1.weight
# features.denseblock4.denselayer1.norm1.bias
# features.denseblock4.denselayer1.conv1.weight
# features.denseblock4.denselayer1.norm2.weight
# features.denseblock4.denselayer1.norm2.bias
# features.denseblock4.denselayer1.conv2.weight
# features.denseblock4.denselayer2.norm1.weight
# features.denseblock4.denselayer2.norm1.bias
# features.denseblock4.denselayer2.conv1.weight
# features.denseblock4.denselayer2.norm2.weight
# features.denseblock4.denselayer2.norm2.bias
# features.denseblock4.denselayer2.conv2.weight
# features.denseblock4.denselayer3.norm1.weight
# features.denseblock4.denselayer3.norm1.bias
# features.denseblock4.denselayer3.conv1.weight
# features.denseblock4.denselayer3.norm2.weight
# features.denseblock4.denselayer3.norm2.bias
# features.denseblock4.denselayer3.conv2.weight
# features.denseblock4.denselayer4.norm1.weight
# features.denseblock4.denselayer4.norm1.bias
# features.denseblock4.denselayer4.conv1.weight
# features.denseblock4.denselayer4.norm2.weight
# features.denseblock4.denselayer4.norm2.bias
# features.denseblock4.denselayer4.conv2.weight
# features.denseblock4.denselayer5.norm1.weight
# features.denseblock4.denselayer5.norm1.bias
# features.denseblock4.denselayer5.conv1.weight
# features.denseblock4.denselayer5.norm2.weight
# features.denseblock4.denselayer5.norm2.bias
# features.denseblock4.denselayer5.conv2.weight
# features.denseblock4.denselayer6.norm1.weight
# features.denseblock4.denselayer6.norm1.bias
# features.denseblock4.denselayer6.conv1.weight
# features.denseblock4.denselayer6.norm2.weight
# features.denseblock4.denselayer6.norm2.bias
# features.denseblock4.denselayer6.conv2.weight
# features.denseblock4.denselayer7.norm1.weight
# features.denseblock4.denselayer7.norm1.bias
# features.denseblock4.denselayer7.conv1.weight
# features.denseblock4.denselayer7.norm2.weight
# features.denseblock4.denselayer7.norm2.bias
# features.denseblock4.denselayer7.conv2.weight
# features.denseblock4.denselayer8.norm1.weight
# features.denseblock4.denselayer8.norm1.bias
# features.denseblock4.denselayer8.conv1.weight
# features.denseblock4.denselayer8.norm2.weight
# features.denseblock4.denselayer8.norm2.bias
# features.denseblock4.denselayer8.conv2.weight
# features.denseblock4.denselayer9.norm1.weight
# features.denseblock4.denselayer9.norm1.bias
# features.denseblock4.denselayer9.conv1.weight
# features.denseblock4.denselayer9.norm2.weight
# features.denseblock4.denselayer9.norm2.bias
# features.denseblock4.denselayer9.conv2.weight
# features.denseblock4.denselayer10.norm1.weight
# features.denseblock4.denselayer10.norm1.bias
# features.denseblock4.denselayer10.conv1.weight
# features.denseblock4.denselayer10.norm2.weight
# features.denseblock4.denselayer10.norm2.bias
# features.denseblock4.denselayer10.conv2.weight
# features.denseblock4.denselayer11.norm1.weight
# features.denseblock4.denselayer11.norm1.bias
# features.denseblock4.denselayer11.conv1.weight
# features.denseblock4.denselayer11.norm2.weight
# features.denseblock4.denselayer11.norm2.bias
# features.denseblock4.denselayer11.conv2.weight
# features.denseblock4.denselayer12.norm1.weight
# features.denseblock4.denselayer12.norm1.bias
# features.denseblock4.denselayer12.conv1.weight
# features.denseblock4.denselayer12.norm2.weight
# features.denseblock4.denselayer12.norm2.bias
# features.denseblock4.denselayer12.conv2.weight
# features.denseblock4.denselayer13.norm1.weight
# features.denseblock4.denselayer13.norm1.bias
# features.denseblock4.denselayer13.conv1.weight
# features.denseblock4.denselayer13.norm2.weight
# features.denseblock4.denselayer13.norm2.bias
# features.denseblock4.denselayer13.conv2.weight
# features.denseblock4.denselayer14.norm1.weight
# features.denseblock4.denselayer14.norm1.bias
# features.denseblock4.denselayer14.conv1.weight
# features.denseblock4.denselayer14.norm2.weight
# features.denseblock4.denselayer14.norm2.bias
# features.denseblock4.denselayer14.conv2.weight
# features.denseblock4.denselayer15.norm1.weight
# features.denseblock4.denselayer15.norm1.bias
# features.denseblock4.denselayer15.conv1.weight
# features.denseblock4.denselayer15.norm2.weight
# features.denseblock4.denselayer15.norm2.bias
# features.denseblock4.denselayer15.conv2.weight
# features.denseblock4.denselayer16.norm1.weight
# features.denseblock4.denselayer16.norm1.bias
# features.denseblock4.denselayer16.conv1.weight
# features.denseblock4.denselayer16.norm2.weight
# features.denseblock4.denselayer16.norm2.bias
# features.denseblock4.denselayer16.conv2.weight
# features.norm5.weight
# features.norm5.bias
# classifier.weight
# classifier.bias
