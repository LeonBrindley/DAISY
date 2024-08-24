import torch
import torch.nn as nn
import torch.optim as optim
import fine_tune
import dropout

# INPUT SIZE: 384x384

class DeepVergeCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepVergeCNN, self).__init__()
        # 1st Convolution block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=1, padding=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2nd Convolution block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3rd Convolution block (repeated twice)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 5th Convolution block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool5a = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5b = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Final Convolution block
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 6400)
        self.fc2 = nn.Linear(6400, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.fc4 = nn.Linear(1000, num_classes)
        
        # Sigmoid activation for multi-label classification
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):    
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv5(x))
        x = self.pool5b(self.pool5a(x))
        x = self.relu(self.conv6(x))
        x = self.pool6(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
def load(dropout_p):
    new_num_classes = 4
    model = DeepVergeCNN(new_num_classes)
    if dropout_p is not None:
        model = dropout.add_dropout(model, "fc", dropout_p)
    return model

# experiment params:
# learning_rate: 0.001, 0.0001, 0.00001
# epochs: 30
# patience: 5
def train_deepverge(
    train_loader, val_loader, weight_decay, dropout_p,
    learning_rate=0.001, epochs=10, patience=5):

    model = load(dropout_p)

    # unfreeze all weights
    fine_tune.unfreeze_all_weights(model)

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