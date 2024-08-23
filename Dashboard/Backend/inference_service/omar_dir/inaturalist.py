import torch
from torchvision.models import resnet50
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
# import keras
import os
import inference_service.omar_dir.dropout as dropout
# import inference_service.omar_dir.fine_tune

class PerceptronMultiLabelResNet(torch.nn.Module):
    def __init__(self, original_model, num_classes):
        super(PerceptronMultiLabelResNet, self).__init__()
        self.resnet = original_model
        self.resnet.fc = nn.Linear(in_features=2048, out_features=512)  # Adjust the size as needed
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class MultiLabelResNet(torch.nn.Module):
    def __init__(self, original_model, num_classes):
        super(MultiLabelResNet, self).__init__()
        self.resnet = original_model
        self.resnet.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x

def load_iNaturalist_model():

  model = resnet50()
#   num_iNaturalist_classes = 10000 # 10,000 classes including all species
#   model.fc = torch.nn.Linear(in_features=2048, out_features=num_iNaturalist_classes)

#   weights_file_path = "/rds/user/omsst2/hpc-work/gp/cvpr21_newt_pretrained_models/pt/inat2021_supervised_large.pth.tar"
#   checkpoint = torch.load(weights_file_path, map_location=torch.device('cpu'))
#   model.load_state_dict(checkpoint['state_dict'])

  return model

def load(use_extra_perceptron, dropout_p):
    model = load_iNaturalist_model()

    new_num_classes = 4  # {Clover, Grass, Dung, Soil}

    model = PerceptronMultiLabelResNet(model, new_num_classes) if use_extra_perceptron else MultiLabelResNet(model, new_num_classes)
    if dropout_p is not None:
        # note: already got dropout in its classification layer
        model = dropout.add_dropout(model, "fc", dropout_p)
    return model


# experiment params:
# unfreeze_layers: fc, layer4, layer3
# use_extra_perceptron: True, False
# learning_rate: 0.001, 0.0001, 0.00001, 0.000001, 0.0000001
# epochs: 30
# patience: 5
# def fine_tune_inaturalist(
#     train_loader, val_loader, use_extra_perceptron, unfreeze_layers,
#     learning_rate=0.001, epochs=10, patience=5):

#     model = load(use_extra_perceptron)    

#     # freeze all weights
#     fine_tune.freeze_weights(model)

#     # unfreeze final layers for finetuning
#     fine_tune.unfreeze_weights(model, unfreeze_layers)

#     # Define loss function and optimiser
#     loss_function = nn.BCELoss()  # Binary cross-entropy loss for one-hot labels
#     optimiser = optim.Adam(model.parameters(), lr=learning_rate)  # Adjust learning rate as needed

#     # Detect if GPU is available
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Fine-tune the model
#     model, train_stats, val_stats = fine_tune.fine_tune_model(
#         model,
#         device,
#         train_loader, 
#         val_loader, 
#         optimiser, 
#         loss_function, 
#         num_epochs=epochs,
#         patience=patience)


#     return model, train_stats, val_stats


# MultiLabelResNet(
#   (resnet): ResNet(
#     (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#     (layer1): Sequential(
#       (0): Bottleneck(
#         (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (downsample): Sequential(
#           (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): Bottleneck(
#         (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (2): Bottleneck(
#         (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#     )
#     (layer2): Sequential(
#       (0): Bottleneck(
#         (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (downsample): Sequential(
#           (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): Bottleneck(
#         (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (2): Bottleneck(
#         (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (3): Bottleneck(
#         (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#     )
#     (layer3): Sequential(
#       (0): Bottleneck(
#         (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (downsample): Sequential(
#           (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (2): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (3): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (4): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (5): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#     )
#     (layer4): Sequential(
#       (0): Bottleneck(
#         (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (downsample): Sequential(
#           (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): Bottleneck(
#         (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (2): Bottleneck(
#         (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#     )
#     (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#     (fc): Linear(in_features=2048, out_features=4, bias=True)
#   )
#   (sigmoid): Sigmoid()
# )
# PARAMS
# resnet.conv1.weight
# resnet.bn1.weight
# resnet.bn1.bias
# resnet.layer1.0.conv1.weight
# resnet.layer1.0.bn1.weight
# resnet.layer1.0.bn1.bias
# resnet.layer1.0.conv2.weight
# resnet.layer1.0.bn2.weight
# resnet.layer1.0.bn2.bias
# resnet.layer1.0.conv3.weight
# resnet.layer1.0.bn3.weight
# resnet.layer1.0.bn3.bias
# resnet.layer1.0.downsample.0.weight
# resnet.layer1.0.downsample.1.weight
# resnet.layer1.0.downsample.1.bias
# resnet.layer1.1.conv1.weight
# resnet.layer1.1.bn1.weight
# resnet.layer1.1.bn1.bias
# resnet.layer1.1.conv2.weight
# resnet.layer1.1.bn2.weight
# resnet.layer1.1.bn2.bias
# resnet.layer1.1.conv3.weight
# resnet.layer1.1.bn3.weight
# resnet.layer1.1.bn3.bias
# resnet.layer1.2.conv1.weight
# resnet.layer1.2.bn1.weight
# resnet.layer1.2.bn1.bias
# resnet.layer1.2.conv2.weight
# resnet.layer1.2.bn2.weight
# resnet.layer1.2.bn2.bias
# resnet.layer1.2.conv3.weight
# resnet.layer1.2.bn3.weight
# resnet.layer1.2.bn3.bias
# resnet.layer2.0.conv1.weight
# resnet.layer2.0.bn1.weight
# resnet.layer2.0.bn1.bias
# resnet.layer2.0.conv2.weight
# resnet.layer2.0.bn2.weight
# resnet.layer2.0.bn2.bias
# resnet.layer2.0.conv3.weight
# resnet.layer2.0.bn3.weight
# resnet.layer2.0.bn3.bias
# resnet.layer2.0.downsample.0.weight
# resnet.layer2.0.downsample.1.weight
# resnet.layer2.0.downsample.1.bias
# resnet.layer2.1.conv1.weight
# resnet.layer2.1.bn1.weight
# resnet.layer2.1.bn1.bias
# resnet.layer2.1.conv2.weight
# resnet.layer2.1.bn2.weight
# resnet.layer2.1.bn2.bias
# resnet.layer2.1.conv3.weight
# resnet.layer2.1.bn3.weight
# resnet.layer2.1.bn3.bias
# resnet.layer2.2.conv1.weight
# resnet.layer2.2.bn1.weight
# resnet.layer2.2.bn1.bias
# resnet.layer2.2.conv2.weight
# resnet.layer2.2.bn2.weight
# resnet.layer2.2.bn2.bias
# resnet.layer2.2.conv3.weight
# resnet.layer2.2.bn3.weight
# resnet.layer2.2.bn3.bias
# resnet.layer2.3.conv1.weight
# resnet.layer2.3.bn1.weight
# resnet.layer2.3.bn1.bias
# resnet.layer2.3.conv2.weight
# resnet.layer2.3.bn2.weight
# resnet.layer2.3.bn2.bias
# resnet.layer2.3.conv3.weight
# resnet.layer2.3.bn3.weight
# resnet.layer2.3.bn3.bias
# resnet.layer3.0.conv1.weight
# resnet.layer3.0.bn1.weight
# resnet.layer3.0.bn1.bias
# resnet.layer3.0.conv2.weight
# resnet.layer3.0.bn2.weight
# resnet.layer3.0.bn2.bias
# resnet.layer3.0.conv3.weight
# resnet.layer3.0.bn3.weight
# resnet.layer3.0.bn3.bias
# resnet.layer3.0.downsample.0.weight
# resnet.layer3.0.downsample.1.weight
# resnet.layer3.0.downsample.1.bias
# resnet.layer3.1.conv1.weight
# resnet.layer3.1.bn1.weight
# resnet.layer3.1.bn1.bias
# resnet.layer3.1.conv2.weight
# resnet.layer3.1.bn2.weight
# resnet.layer3.1.bn2.bias
# resnet.layer3.1.conv3.weight
# resnet.layer3.1.bn3.weight
# resnet.layer3.1.bn3.bias
# resnet.layer3.2.conv1.weight
# resnet.layer3.2.bn1.weight
# resnet.layer3.2.bn1.bias
# resnet.layer3.2.conv2.weight
# resnet.layer3.2.bn2.weight
# resnet.layer3.2.bn2.bias
# resnet.layer3.2.conv3.weight
# resnet.layer3.2.bn3.weight
# resnet.layer3.2.bn3.bias
# resnet.layer3.3.conv1.weight
# resnet.layer3.3.bn1.weight
# resnet.layer3.3.bn1.bias
# resnet.layer3.3.conv2.weight
# resnet.layer3.3.bn2.weight
# resnet.layer3.3.bn2.bias
# resnet.layer3.3.conv3.weight
# resnet.layer3.3.bn3.weight
# resnet.layer3.3.bn3.bias
# resnet.layer3.4.conv1.weight
# resnet.layer3.4.bn1.weight
# resnet.layer3.4.bn1.bias
# resnet.layer3.4.conv2.weight
# resnet.layer3.4.bn2.weight
# resnet.layer3.4.bn2.bias
# resnet.layer3.4.conv3.weight
# resnet.layer3.4.bn3.weight
# resnet.layer3.4.bn3.bias
# resnet.layer3.5.conv1.weight
# resnet.layer3.5.bn1.weight
# resnet.layer3.5.bn1.bias
# resnet.layer3.5.conv2.weight
# resnet.layer3.5.bn2.weight
# resnet.layer3.5.bn2.bias
# resnet.layer3.5.conv3.weight
# resnet.layer3.5.bn3.weight
# resnet.layer3.5.bn3.bias
# resnet.layer4.0.conv1.weight
# resnet.layer4.0.bn1.weight
# resnet.layer4.0.bn1.bias
# resnet.layer4.0.conv2.weight
# resnet.layer4.0.bn2.weight
# resnet.layer4.0.bn2.bias
# resnet.layer4.0.conv3.weight
# resnet.layer4.0.bn3.weight
# resnet.layer4.0.bn3.bias
# resnet.layer4.0.downsample.0.weight
# resnet.layer4.0.downsample.1.weight
# resnet.layer4.0.downsample.1.bias
# resnet.layer4.1.conv1.weight
# resnet.layer4.1.bn1.weight
# resnet.layer4.1.bn1.bias
# resnet.layer4.1.conv2.weight
# resnet.layer4.1.bn2.weight
# resnet.layer4.1.bn2.bias
# resnet.layer4.1.conv3.weight
# resnet.layer4.1.bn3.weight
# resnet.layer4.1.bn3.bias
# resnet.layer4.2.conv1.weight
# resnet.layer4.2.bn1.weight
# resnet.layer4.2.bn1.bias
# resnet.layer4.2.conv2.weight
# resnet.layer4.2.bn2.weight
# resnet.layer4.2.bn2.bias
# resnet.layer4.2.conv3.weight
# resnet.layer4.2.bn3.weight
# resnet.layer4.2.bn3.bias
# resnet.fc.weight
# resnet.fc.bias