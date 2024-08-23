import torch
from torchvision.models import inception_v3
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
# import keras
import os
# import inference_service.omar_dir.fine_tune

class PerceptronMultiLabelInceptionV3(nn.Module):
    def __init__(self, original_model, num_classes):
        super(PerceptronMultiLabelInceptionV3, self).__init__()
        self.inception = original_model
        self.inception.fc = nn.Linear(in_features=2048, out_features=512)
        self.inception.AuxLogits.fc = nn.Linear(in_features=768, out_features=512)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.training and self.inception.aux_logits:
            x, aux = self.inception(x)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.sigmoid(x)
            aux = self.relu(self.fc1(aux))
            aux = self.fc2(aux)
            aux = self.sigmoid(aux)
            return x, aux
        else:
            x = self.inception(x)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x


class MultiLabelInceptionV3(nn.Module):
    def __init__(self, original_model, num_classes):
        super(MultiLabelInceptionV3, self).__init__()
        self.inception = original_model
        self.inception.fc = nn.Linear(in_features=2048, out_features=num_classes)
        self.inception.AuxLogits.fc = nn.Linear(in_features=768, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.training and self.inception.aux_logits:
            x, aux = self.inception(x)
            x = self.sigmoid(x)
            aux = self.sigmoid(aux)
            return x, aux
        else:
            x = self.inception(x)
            x = self.sigmoid(x)
            return x

def load_inceptionv3_model():
  return inception_v3(pretrained=True, aux_logits=True) # InceptionV3 requires aux_logits=True for training

# experiment params:
# unfreeze_layers: inception.fc, inception.Mixed_7c, inception.Mixed_7b, inception.Mixed_7a, inception.AuxLogits, inception.Mixed_6e, inception.Mixed_6d
# use_extra_perceptron: True, False
# learning_rate: 0.001, 0.0001, 0.00001, 0.000001, 0.0000001
# epochs: 30
# patience: 5
def fine_tune_inceptionv3(
    train_loader, val_loader, use_extra_perceptron, unfreeze_layers,
    learning_rate=0.001, epochs=10, patience=5):

    model = load_inceptionv3_model()

    new_num_classes = 4  # {Clover, Grass, Dung, Soil}

    model = PerceptronMultiLabelInceptionV3(model, new_num_classes) if use_extra_perceptron else MultiLabelInceptionV3(model, new_num_classes)

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
    model, train_stats, val_stats = fine_tune.fine_tune_model_aux(
        model,
        device,
        train_loader, 
        val_loader, 
        optimiser, 
        loss_function, 
        num_epochs=epochs,
        patience=patience)


    return model, train_stats, val_stats


# PerceptronMultiLabelInceptionV3(
#   (inception): Inception3(
#     (Conv2d_1a_3x3): BasicConv2d(
#       (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
#       (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (Conv2d_2a_3x3): BasicConv2d(
#       (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (Conv2d_2b_3x3): BasicConv2d(
#       (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (maxpool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (Conv2d_3b_1x1): BasicConv2d(
#       (conv): Conv2d(64, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (Conv2d_4a_3x3): BasicConv2d(
#       (conv): Conv2d(80, 192, kernel_size=(3, 3), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (maxpool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (Mixed_5b): InceptionA(
#       (branch1x1): BasicConv2d(
#         (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch5x5_1): BasicConv2d(
#         (conv): Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch5x5_2): BasicConv2d(
#         (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_1): BasicConv2d(
#         (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_2): BasicConv2d(
#         (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_3): BasicConv2d(
#         (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch_pool): BasicConv2d(
#         (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (Mixed_5c): InceptionA(
#       (branch1x1): BasicConv2d(
#         (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch5x5_1): BasicConv2d(
#         (conv): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch5x5_2): BasicConv2d(
#         (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_1): BasicConv2d(
#         (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_2): BasicConv2d(
#         (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_3): BasicConv2d(
#         (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch_pool): BasicConv2d(
#         (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (Mixed_5d): InceptionA(
#       (branch1x1): BasicConv2d(
#         (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch5x5_1): BasicConv2d(
#         (conv): Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch5x5_2): BasicConv2d(
#         (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_1): BasicConv2d(
#         (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_2): BasicConv2d(
#         (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_3): BasicConv2d(
#         (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch_pool): BasicConv2d(
#         (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (Mixed_6a): InceptionB(
#       (branch3x3): BasicConv2d(
#         (conv): Conv2d(288, 384, kernel_size=(3, 3), stride=(2, 2), bias=False)
#         (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_1): BasicConv2d(
#         (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_2): BasicConv2d(
#         (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_3): BasicConv2d(
#         (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), bias=False)
#         (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (Mixed_6b): InceptionC(
#       (branch1x1): BasicConv2d(
#         (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7_1): BasicConv2d(
#         (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7_2): BasicConv2d(
#         (conv): Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7_3): BasicConv2d(
#         (conv): Conv2d(128, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_1): BasicConv2d(
#         (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_2): BasicConv2d(
#         (conv): Conv2d(128, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_3): BasicConv2d(
#         (conv): Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_4): BasicConv2d(
#         (conv): Conv2d(128, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_5): BasicConv2d(
#         (conv): Conv2d(128, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch_pool): BasicConv2d(
#         (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (Mixed_6c): InceptionC(
#       (branch1x1): BasicConv2d(
#         (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7_1): BasicConv2d(
#         (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7_2): BasicConv2d(
#         (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#         (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7_3): BasicConv2d(
#         (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_1): BasicConv2d(
#         (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_2): BasicConv2d(
#         (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#         (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_3): BasicConv2d(
#         (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#         (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_4): BasicConv2d(
#         (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#         (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_5): BasicConv2d(
#         (conv): Conv2d(160, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch_pool): BasicConv2d(
#         (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (Mixed_6d): InceptionC(
#       (branch1x1): BasicConv2d(
#         (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7_1): BasicConv2d(
#         (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7_2): BasicConv2d(
#         (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#         (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7_3): BasicConv2d(
#         (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_1): BasicConv2d(
#         (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_2): BasicConv2d(
#         (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#         (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_3): BasicConv2d(
#         (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#         (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_4): BasicConv2d(
#         (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#         (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_5): BasicConv2d(
#         (conv): Conv2d(160, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch_pool): BasicConv2d(
#         (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (Mixed_6e): InceptionC(
#       (branch1x1): BasicConv2d(
#         (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7_1): BasicConv2d(
#         (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7_2): BasicConv2d(
#         (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7_3): BasicConv2d(
#         (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_1): BasicConv2d(
#         (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_2): BasicConv2d(
#         (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_3): BasicConv2d(
#         (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_4): BasicConv2d(
#         (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7dbl_5): BasicConv2d(
#         (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch_pool): BasicConv2d(
#         (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (AuxLogits): InceptionAux(
#       (conv0): BasicConv2d(
#         (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (conv1): BasicConv2d(
#         (conv): Conv2d(128, 768, kernel_size=(5, 5), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(768, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (fc): Linear(in_features=768, out_features=512, bias=True)
#     )
#     (Mixed_7a): InceptionD(
#       (branch3x3_1): BasicConv2d(
#         (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3_2): BasicConv2d(
#         (conv): Conv2d(192, 320, kernel_size=(3, 3), stride=(2, 2), bias=False)
#         (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7x3_1): BasicConv2d(
#         (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7x3_2): BasicConv2d(
#         (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7x3_3): BasicConv2d(
#         (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch7x7x3_4): BasicConv2d(
#         (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (Mixed_7b): InceptionE(
#       (branch1x1): BasicConv2d(
#         (conv): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3_1): BasicConv2d(
#         (conv): Conv2d(1280, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3_2a): BasicConv2d(
#         (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
#         (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3_2b): BasicConv2d(
#         (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
#         (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_1): BasicConv2d(
#         (conv): Conv2d(1280, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(448, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_2): BasicConv2d(
#         (conv): Conv2d(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_3a): BasicConv2d(
#         (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
#         (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_3b): BasicConv2d(
#         (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
#         (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch_pool): BasicConv2d(
#         (conv): Conv2d(1280, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (Mixed_7c): InceptionE(
#       (branch1x1): BasicConv2d(
#         (conv): Conv2d(2048, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3_1): BasicConv2d(
#         (conv): Conv2d(2048, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3_2a): BasicConv2d(
#         (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
#         (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3_2b): BasicConv2d(
#         (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
#         (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_1): BasicConv2d(
#         (conv): Conv2d(2048, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(448, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_2): BasicConv2d(
#         (conv): Conv2d(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_3a): BasicConv2d(
#         (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
#         (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch3x3dbl_3b): BasicConv2d(
#         (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
#         (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (branch_pool): BasicConv2d(
#         (conv): Conv2d(2048, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#     (dropout): Dropout(p=0.5, inplace=False)
#     (fc): Linear(in_features=2048, out_features=512, bias=True)
#   )
#   (fc1): Linear(in_features=512, out_features=256, bias=True)
#   (fc2): Linear(in_features=256, out_features=4, bias=True)
#   (relu): ReLU()
#   (sigmoid): Sigmoid()
# )
# PARAMS
# inception.Conv2d_1a_3x3.conv.weight
# inception.Conv2d_1a_3x3.bn.weight
# inception.Conv2d_1a_3x3.bn.bias
# inception.Conv2d_2a_3x3.conv.weight
# inception.Conv2d_2a_3x3.bn.weight
# inception.Conv2d_2a_3x3.bn.bias
# inception.Conv2d_2b_3x3.conv.weight
# inception.Conv2d_2b_3x3.bn.weight
# inception.Conv2d_2b_3x3.bn.bias
# inception.Conv2d_3b_1x1.conv.weight
# inception.Conv2d_3b_1x1.bn.weight
# inception.Conv2d_3b_1x1.bn.bias
# inception.Conv2d_4a_3x3.conv.weight
# inception.Conv2d_4a_3x3.bn.weight
# inception.Conv2d_4a_3x3.bn.bias
# inception.Mixed_5b.branch1x1.conv.weight
# inception.Mixed_5b.branch1x1.bn.weight
# inception.Mixed_5b.branch1x1.bn.bias
# inception.Mixed_5b.branch5x5_1.conv.weight
# inception.Mixed_5b.branch5x5_1.bn.weight
# inception.Mixed_5b.branch5x5_1.bn.bias
# inception.Mixed_5b.branch5x5_2.conv.weight
# inception.Mixed_5b.branch5x5_2.bn.weight
# inception.Mixed_5b.branch5x5_2.bn.bias
# inception.Mixed_5b.branch3x3dbl_1.conv.weight
# inception.Mixed_5b.branch3x3dbl_1.bn.weight
# inception.Mixed_5b.branch3x3dbl_1.bn.bias
# inception.Mixed_5b.branch3x3dbl_2.conv.weight
# inception.Mixed_5b.branch3x3dbl_2.bn.weight
# inception.Mixed_5b.branch3x3dbl_2.bn.bias
# inception.Mixed_5b.branch3x3dbl_3.conv.weight
# inception.Mixed_5b.branch3x3dbl_3.bn.weight
# inception.Mixed_5b.branch3x3dbl_3.bn.bias
# inception.Mixed_5b.branch_pool.conv.weight
# inception.Mixed_5b.branch_pool.bn.weight
# inception.Mixed_5b.branch_pool.bn.bias
# inception.Mixed_5c.branch1x1.conv.weight
# inception.Mixed_5c.branch1x1.bn.weight
# inception.Mixed_5c.branch1x1.bn.bias
# inception.Mixed_5c.branch5x5_1.conv.weight
# inception.Mixed_5c.branch5x5_1.bn.weight
# inception.Mixed_5c.branch5x5_1.bn.bias
# inception.Mixed_5c.branch5x5_2.conv.weight
# inception.Mixed_5c.branch5x5_2.bn.weight
# inception.Mixed_5c.branch5x5_2.bn.bias
# inception.Mixed_5c.branch3x3dbl_1.conv.weight
# inception.Mixed_5c.branch3x3dbl_1.bn.weight
# inception.Mixed_5c.branch3x3dbl_1.bn.bias
# inception.Mixed_5c.branch3x3dbl_2.conv.weight
# inception.Mixed_5c.branch3x3dbl_2.bn.weight
# inception.Mixed_5c.branch3x3dbl_2.bn.bias
# inception.Mixed_5c.branch3x3dbl_3.conv.weight
# inception.Mixed_5c.branch3x3dbl_3.bn.weight
# inception.Mixed_5c.branch3x3dbl_3.bn.bias
# inception.Mixed_5c.branch_pool.conv.weight
# inception.Mixed_5c.branch_pool.bn.weight
# inception.Mixed_5c.branch_pool.bn.bias
# inception.Mixed_5d.branch1x1.conv.weight
# inception.Mixed_5d.branch1x1.bn.weight
# inception.Mixed_5d.branch1x1.bn.bias
# inception.Mixed_5d.branch5x5_1.conv.weight
# inception.Mixed_5d.branch5x5_1.bn.weight
# inception.Mixed_5d.branch5x5_1.bn.bias
# inception.Mixed_5d.branch5x5_2.conv.weight
# inception.Mixed_5d.branch5x5_2.bn.weight
# inception.Mixed_5d.branch5x5_2.bn.bias
# inception.Mixed_5d.branch3x3dbl_1.conv.weight
# inception.Mixed_5d.branch3x3dbl_1.bn.weight
# inception.Mixed_5d.branch3x3dbl_1.bn.bias
# inception.Mixed_5d.branch3x3dbl_2.conv.weight
# inception.Mixed_5d.branch3x3dbl_2.bn.weight
# inception.Mixed_5d.branch3x3dbl_2.bn.bias
# inception.Mixed_5d.branch3x3dbl_3.conv.weight
# inception.Mixed_5d.branch3x3dbl_3.bn.weight
# inception.Mixed_5d.branch3x3dbl_3.bn.bias
# inception.Mixed_5d.branch_pool.conv.weight
# inception.Mixed_5d.branch_pool.bn.weight
# inception.Mixed_5d.branch_pool.bn.bias
# inception.Mixed_6a.branch3x3.conv.weight
# inception.Mixed_6a.branch3x3.bn.weight
# inception.Mixed_6a.branch3x3.bn.bias
# inception.Mixed_6a.branch3x3dbl_1.conv.weight
# inception.Mixed_6a.branch3x3dbl_1.bn.weight
# inception.Mixed_6a.branch3x3dbl_1.bn.bias
# inception.Mixed_6a.branch3x3dbl_2.conv.weight
# inception.Mixed_6a.branch3x3dbl_2.bn.weight
# inception.Mixed_6a.branch3x3dbl_2.bn.bias
# inception.Mixed_6a.branch3x3dbl_3.conv.weight
# inception.Mixed_6a.branch3x3dbl_3.bn.weight
# inception.Mixed_6a.branch3x3dbl_3.bn.bias
# inception.Mixed_6b.branch1x1.conv.weight
# inception.Mixed_6b.branch1x1.bn.weight
# inception.Mixed_6b.branch1x1.bn.bias
# inception.Mixed_6b.branch7x7_1.conv.weight
# inception.Mixed_6b.branch7x7_1.bn.weight
# inception.Mixed_6b.branch7x7_1.bn.bias
# inception.Mixed_6b.branch7x7_2.conv.weight
# inception.Mixed_6b.branch7x7_2.bn.weight
# inception.Mixed_6b.branch7x7_2.bn.bias
# inception.Mixed_6b.branch7x7_3.conv.weight
# inception.Mixed_6b.branch7x7_3.bn.weight
# inception.Mixed_6b.branch7x7_3.bn.bias
# inception.Mixed_6b.branch7x7dbl_1.conv.weight
# inception.Mixed_6b.branch7x7dbl_1.bn.weight
# inception.Mixed_6b.branch7x7dbl_1.bn.bias
# inception.Mixed_6b.branch7x7dbl_2.conv.weight
# inception.Mixed_6b.branch7x7dbl_2.bn.weight
# inception.Mixed_6b.branch7x7dbl_2.bn.bias
# inception.Mixed_6b.branch7x7dbl_3.conv.weight
# inception.Mixed_6b.branch7x7dbl_3.bn.weight
# inception.Mixed_6b.branch7x7dbl_3.bn.bias
# inception.Mixed_6b.branch7x7dbl_4.conv.weight
# inception.Mixed_6b.branch7x7dbl_4.bn.weight
# inception.Mixed_6b.branch7x7dbl_4.bn.bias
# inception.Mixed_6b.branch7x7dbl_5.conv.weight
# inception.Mixed_6b.branch7x7dbl_5.bn.weight
# inception.Mixed_6b.branch7x7dbl_5.bn.bias
# inception.Mixed_6b.branch_pool.conv.weight
# inception.Mixed_6b.branch_pool.bn.weight
# inception.Mixed_6b.branch_pool.bn.bias
# inception.Mixed_6c.branch1x1.conv.weight
# inception.Mixed_6c.branch1x1.bn.weight
# inception.Mixed_6c.branch1x1.bn.bias
# inception.Mixed_6c.branch7x7_1.conv.weight
# inception.Mixed_6c.branch7x7_1.bn.weight
# inception.Mixed_6c.branch7x7_1.bn.bias
# inception.Mixed_6c.branch7x7_2.conv.weight
# inception.Mixed_6c.branch7x7_2.bn.weight
# inception.Mixed_6c.branch7x7_2.bn.bias
# inception.Mixed_6c.branch7x7_3.conv.weight
# inception.Mixed_6c.branch7x7_3.bn.weight
# inception.Mixed_6c.branch7x7_3.bn.bias
# inception.Mixed_6c.branch7x7dbl_1.conv.weight
# inception.Mixed_6c.branch7x7dbl_1.bn.weight
# inception.Mixed_6c.branch7x7dbl_1.bn.bias
# inception.Mixed_6c.branch7x7dbl_2.conv.weight
# inception.Mixed_6c.branch7x7dbl_2.bn.weight
# inception.Mixed_6c.branch7x7dbl_2.bn.bias
# inception.Mixed_6c.branch7x7dbl_3.conv.weight
# inception.Mixed_6c.branch7x7dbl_3.bn.weight
# inception.Mixed_6c.branch7x7dbl_3.bn.bias
# inception.Mixed_6c.branch7x7dbl_4.conv.weight
# inception.Mixed_6c.branch7x7dbl_4.bn.weight
# inception.Mixed_6c.branch7x7dbl_4.bn.bias
# inception.Mixed_6c.branch7x7dbl_5.conv.weight
# inception.Mixed_6c.branch7x7dbl_5.bn.weight
# inception.Mixed_6c.branch7x7dbl_5.bn.bias
# inception.Mixed_6c.branch_pool.conv.weight
# inception.Mixed_6c.branch_pool.bn.weight
# inception.Mixed_6c.branch_pool.bn.bias
# inception.Mixed_6d.branch1x1.conv.weight
# inception.Mixed_6d.branch1x1.bn.weight
# inception.Mixed_6d.branch1x1.bn.bias
# inception.Mixed_6d.branch7x7_1.conv.weight
# inception.Mixed_6d.branch7x7_1.bn.weight
# inception.Mixed_6d.branch7x7_1.bn.bias
# inception.Mixed_6d.branch7x7_2.conv.weight
# inception.Mixed_6d.branch7x7_2.bn.weight
# inception.Mixed_6d.branch7x7_2.bn.bias
# inception.Mixed_6d.branch7x7_3.conv.weight
# inception.Mixed_6d.branch7x7_3.bn.weight
# inception.Mixed_6d.branch7x7_3.bn.bias
# inception.Mixed_6d.branch7x7dbl_1.conv.weight
# inception.Mixed_6d.branch7x7dbl_1.bn.weight
# inception.Mixed_6d.branch7x7dbl_1.bn.bias
# inception.Mixed_6d.branch7x7dbl_2.conv.weight
# inception.Mixed_6d.branch7x7dbl_2.bn.weight
# inception.Mixed_6d.branch7x7dbl_2.bn.bias
# inception.Mixed_6d.branch7x7dbl_3.conv.weight
# inception.Mixed_6d.branch7x7dbl_3.bn.weight
# inception.Mixed_6d.branch7x7dbl_3.bn.bias
# inception.Mixed_6d.branch7x7dbl_4.conv.weight
# inception.Mixed_6d.branch7x7dbl_4.bn.weight
# inception.Mixed_6d.branch7x7dbl_4.bn.bias
# inception.Mixed_6d.branch7x7dbl_5.conv.weight
# inception.Mixed_6d.branch7x7dbl_5.bn.weight
# inception.Mixed_6d.branch7x7dbl_5.bn.bias
# inception.Mixed_6d.branch_pool.conv.weight
# inception.Mixed_6d.branch_pool.bn.weight
# inception.Mixed_6d.branch_pool.bn.bias
# inception.Mixed_6e.branch1x1.conv.weight
# inception.Mixed_6e.branch1x1.bn.weight
# inception.Mixed_6e.branch1x1.bn.bias
# inception.Mixed_6e.branch7x7_1.conv.weight
# inception.Mixed_6e.branch7x7_1.bn.weight
# inception.Mixed_6e.branch7x7_1.bn.bias
# inception.Mixed_6e.branch7x7_2.conv.weight
# inception.Mixed_6e.branch7x7_2.bn.weight
# inception.Mixed_6e.branch7x7_2.bn.bias
# inception.Mixed_6e.branch7x7_3.conv.weight
# inception.Mixed_6e.branch7x7_3.bn.weight
# inception.Mixed_6e.branch7x7_3.bn.bias
# inception.Mixed_6e.branch7x7dbl_1.conv.weight
# inception.Mixed_6e.branch7x7dbl_1.bn.weight
# inception.Mixed_6e.branch7x7dbl_1.bn.bias
# inception.Mixed_6e.branch7x7dbl_2.conv.weight
# inception.Mixed_6e.branch7x7dbl_2.bn.weight
# inception.Mixed_6e.branch7x7dbl_2.bn.bias
# inception.Mixed_6e.branch7x7dbl_3.conv.weight
# inception.Mixed_6e.branch7x7dbl_3.bn.weight
# inception.Mixed_6e.branch7x7dbl_3.bn.bias
# inception.Mixed_6e.branch7x7dbl_4.conv.weight
# inception.Mixed_6e.branch7x7dbl_4.bn.weight
# inception.Mixed_6e.branch7x7dbl_4.bn.bias
# inception.Mixed_6e.branch7x7dbl_5.conv.weight
# inception.Mixed_6e.branch7x7dbl_5.bn.weight
# inception.Mixed_6e.branch7x7dbl_5.bn.bias
# inception.Mixed_6e.branch_pool.conv.weight
# inception.Mixed_6e.branch_pool.bn.weight
# inception.Mixed_6e.branch_pool.bn.bias
# inception.AuxLogits.conv0.conv.weight
# inception.AuxLogits.conv0.bn.weight
# inception.AuxLogits.conv0.bn.bias
# inception.AuxLogits.conv1.conv.weight
# inception.AuxLogits.conv1.bn.weight
# inception.AuxLogits.conv1.bn.bias
# inception.AuxLogits.fc.weight
# inception.AuxLogits.fc.bias
# inception.Mixed_7a.branch3x3_1.conv.weight
# inception.Mixed_7a.branch3x3_1.bn.weight
# inception.Mixed_7a.branch3x3_1.bn.bias
# inception.Mixed_7a.branch3x3_2.conv.weight
# inception.Mixed_7a.branch3x3_2.bn.weight
# inception.Mixed_7a.branch3x3_2.bn.bias
# inception.Mixed_7a.branch7x7x3_1.conv.weight
# inception.Mixed_7a.branch7x7x3_1.bn.weight
# inception.Mixed_7a.branch7x7x3_1.bn.bias
# inception.Mixed_7a.branch7x7x3_2.conv.weight
# inception.Mixed_7a.branch7x7x3_2.bn.weight
# inception.Mixed_7a.branch7x7x3_2.bn.bias
# inception.Mixed_7a.branch7x7x3_3.conv.weight
# inception.Mixed_7a.branch7x7x3_3.bn.weight
# inception.Mixed_7a.branch7x7x3_3.bn.bias
# inception.Mixed_7a.branch7x7x3_4.conv.weight
# inception.Mixed_7a.branch7x7x3_4.bn.weight
# inception.Mixed_7a.branch7x7x3_4.bn.bias
# inception.Mixed_7b.branch1x1.conv.weight
# inception.Mixed_7b.branch1x1.bn.weight
# inception.Mixed_7b.branch1x1.bn.bias
# inception.Mixed_7b.branch3x3_1.conv.weight
# inception.Mixed_7b.branch3x3_1.bn.weight
# inception.Mixed_7b.branch3x3_1.bn.bias
# inception.Mixed_7b.branch3x3_2a.conv.weight
# inception.Mixed_7b.branch3x3_2a.bn.weight
# inception.Mixed_7b.branch3x3_2a.bn.bias
# inception.Mixed_7b.branch3x3_2b.conv.weight
# inception.Mixed_7b.branch3x3_2b.bn.weight
# inception.Mixed_7b.branch3x3_2b.bn.bias
# inception.Mixed_7b.branch3x3dbl_1.conv.weight
# inception.Mixed_7b.branch3x3dbl_1.bn.weight
# inception.Mixed_7b.branch3x3dbl_1.bn.bias
# inception.Mixed_7b.branch3x3dbl_2.conv.weight
# inception.Mixed_7b.branch3x3dbl_2.bn.weight
# inception.Mixed_7b.branch3x3dbl_2.bn.bias
# inception.Mixed_7b.branch3x3dbl_3a.conv.weight
# inception.Mixed_7b.branch3x3dbl_3a.bn.weight
# inception.Mixed_7b.branch3x3dbl_3a.bn.bias
# inception.Mixed_7b.branch3x3dbl_3b.conv.weight
# inception.Mixed_7b.branch3x3dbl_3b.bn.weight
# inception.Mixed_7b.branch3x3dbl_3b.bn.bias
# inception.Mixed_7b.branch_pool.conv.weight
# inception.Mixed_7b.branch_pool.bn.weight
# inception.Mixed_7b.branch_pool.bn.bias
# inception.Mixed_7c.branch1x1.conv.weight
# inception.Mixed_7c.branch1x1.bn.weight
# inception.Mixed_7c.branch1x1.bn.bias
# inception.Mixed_7c.branch3x3_1.conv.weight
# inception.Mixed_7c.branch3x3_1.bn.weight
# inception.Mixed_7c.branch3x3_1.bn.bias
# inception.Mixed_7c.branch3x3_2a.conv.weight
# inception.Mixed_7c.branch3x3_2a.bn.weight
# inception.Mixed_7c.branch3x3_2a.bn.bias
# inception.Mixed_7c.branch3x3_2b.conv.weight
# inception.Mixed_7c.branch3x3_2b.bn.weight
# inception.Mixed_7c.branch3x3_2b.bn.bias
# inception.Mixed_7c.branch3x3dbl_1.conv.weight
# inception.Mixed_7c.branch3x3dbl_1.bn.weight
# inception.Mixed_7c.branch3x3dbl_1.bn.bias
# inception.Mixed_7c.branch3x3dbl_2.conv.weight
# inception.Mixed_7c.branch3x3dbl_2.bn.weight
# inception.Mixed_7c.branch3x3dbl_2.bn.bias
# inception.Mixed_7c.branch3x3dbl_3a.conv.weight
# inception.Mixed_7c.branch3x3dbl_3a.bn.weight
# inception.Mixed_7c.branch3x3dbl_3a.bn.bias
# inception.Mixed_7c.branch3x3dbl_3b.conv.weight
# inception.Mixed_7c.branch3x3dbl_3b.bn.weight
# inception.Mixed_7c.branch3x3dbl_3b.bn.bias
# inception.Mixed_7c.branch_pool.conv.weight
# inception.Mixed_7c.branch_pool.bn.weight
# inception.Mixed_7c.branch_pool.bn.bias
# inception.fc.weight
# inception.fc.bias
# fc1.weight
# fc1.bias
# fc2.weight
# fc2.bias