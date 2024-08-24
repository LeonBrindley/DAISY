import torch
import torch.nn as nn
import torch.optim as optim
import fine_tune
from torchvision.transforms import Resize
import dropout

# Helper function to resize input to match the model's required input size
def resize_input(x, target_size):
    resize = Resize(target_size)
    return resize(x)

# Large Ensemble Model: 28 -> 16 -> 8 -> 4
class LargeEnsembleModel(nn.Module):
    def __init__(self, models, num_classes, input_sizes):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.input_sizes = input_sizes
        self.oe_fc1 = nn.Linear(len(models) * num_classes, 16)
        self.oe_fc2 = nn.Linear(16, 8)
        self.oe_classifier = nn.Linear(8, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs = []
        for i, model in enumerate(self.models):
            resized_input = resize_input(x, self.input_sizes[i])
            output = model(resized_input)
            if isinstance(output, tuple):
                output = output[0]
            outputs.append(output)
        
        x = torch.cat(outputs, dim=1)
        x = self.relu(self.oe_fc1(x))
        x = self.relu(self.oe_fc2(x))
        out = self.oe_classifier(x)
        out = self.sigmoid(out)
        return out


# Medium Ensemble Model: 28 -> 16 -> 4
class MediumEnsembleModel(nn.Module):
    def __init__(self, models, num_classes, input_sizes):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.input_sizes = input_sizes
        self.oe_fc1 = nn.Linear(len(models) * num_classes, 16)
        self.oe_classifier = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs = []
        for i, model in enumerate(self.models):
            resized_input = resize_input(x, self.input_sizes[i])
            output = model(resized_input)
            if isinstance(output, tuple):
                output = output[0]
            outputs.append(output)
        
        x = torch.cat(outputs, dim=1)
        x = self.relu(self.oe_fc1(x))
        out = self.oe_classifier(x)
        return out


# Small Ensemble Model: 28 -> 4
class SmallEnsembleModel(nn.Module):
    def __init__(self, models, num_classes, input_sizes):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.input_sizes = input_sizes
        self.oe_classifier = nn.Linear(len(models) * num_classes, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs = []
        for i, model in enumerate(self.models):
            resized_input = resize_input(x, self.input_sizes[i])
            output = model(resized_input)
            if isinstance(output, tuple):
                output = output[0]
            outputs.append(output)
        
        x = torch.cat(outputs, dim=1)
        out = self.oe_classifier(x)
        return out

# size = 0: small, 1: medium, 2: large
def train_ensemble(
    train_loader, val_loader, models, input_sizes, size, weight_decay, dropout_p,
    learning_rate=0.001, epochs=10, patience=5):

    # Freeze base models
    for base_model in models:
        fine_tune.freeze_weights(base_model)

    num_classes = 4  # {Clover, Grass, Dung, Soil}

    if size == 0: 
        model = SmallEnsembleModel(models, num_classes, input_sizes)
    elif size == 1: 
        model = MediumEnsembleModel(models, num_classes, input_sizes)
    else:
        model = LargeEnsembleModel(models, num_classes, input_sizes)

    if dropout_p is not None:
        model = dropout.add_dropout(model, "oe_fc", dropout_p)
        model = dropout.add_dropout(model, "oe_classifier", dropout_p)

    # Define loss function and optimiser
    loss_function = nn.BCEWithLogitsLoss()
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