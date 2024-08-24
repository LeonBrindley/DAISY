import torch
from torchvision.models import alexnet
import torch.nn as nn
import torch.optim as optim
import fine_tune
import dropout

class PerceptronMultiLabelAlexNet(torch.nn.Module):
    def __init__(self, original_model, num_classes):
        super(PerceptronMultiLabelAlexNet, self).__init__()
        self.alexnet = original_model
        self.alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=512)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.alexnet(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MultiLabelAlexNet(nn.Module):
    def __init__(self, original_model, num_classes):
        super(MultiLabelAlexNet, self).__init__()
        self.alexnet = original_model
        self.alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.alexnet(x)
        return x

def load_AlexNet_model():
  return alexnet(pretrained=True)

def load(use_extra_perceptron, dropout_p):
    model = load_AlexNet_model()
    new_num_classes = 4  # {Clover, Grass, Dung, Soil}
    model = PerceptronMultiLabelAlexNet(model, new_num_classes) if use_extra_perceptron else MultiLabelAlexNet(model, new_num_classes)
    if dropout_p is not None:
        # note: already got dropout in its classification layer
        model = dropout.add_dropout(model, "fc", dropout_p)

    return model

# experiment params:
# unfreeze_layers: classifier.6, classifier.4, classifier.1, features.10, features.8
# use_extra_perceptron: True, False
# learning_rate: 0.001, 0.0001, 0.00001, 0.000001, 0.0000001
# epochs: 30
# patience: 5
def fine_tune_alexnet(
    train_loader, val_loader, use_extra_perceptron, unfreeze_layers, weight_decay, dropout_p,
    learning_rate=0.001, epochs=10, patience=5):

    model = load(use_extra_perceptron, dropout_p)

    # freeze all weights
    fine_tune.freeze_weights(model)

    # unfreeze final layers for finetuning
    fine_tune.unfreeze_weights(model, unfreeze_layers)

    # Define loss function and optimiser
    loss_function = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for one-hot labels
    if weight_decay is None:
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimiser = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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


# AlexNet(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
#     (1): ReLU(inplace=True)
#     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (4): ReLU(inplace=True)
#     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): ReLU(inplace=True)
#     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): ReLU(inplace=True)
#     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
#   (classifier): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Linear(in_features=9216, out_features=4096, bias=True)
#     (2): ReLU(inplace=True)
#     (3): Dropout(p=0.5, inplace=False)
#     (4): Linear(in_features=4096, out_features=4096, bias=True)
#     (5): ReLU(inplace=True)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )
# PARAMS
# features.0.weight
# features.0.bias
# features.3.weight
# features.3.bias
# features.6.weight
# features.6.bias
# features.8.weight
# features.8.bias
# features.10.weight
# features.10.bias
# classifier.1.weight
# classifier.1.bias
# classifier.4.weight
# classifier.4.bias
# classifier.6.weight
# classifier.6.bias