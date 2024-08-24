import torch
import torch.nn as nn
import torch.optim as optim
import fine_tune
import dropout

# INPUT SIZE: 224x224

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        # 1st Convolutional Layer followed by ReLU
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        # Pooling Layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        # Fully connected layer with ReLU
        self.fc1 = nn.Linear(32 * 112 * 112, 128)
        self.relu2 = nn.ReLU()
        
        # Output fully connected layer
        self.fc2 = nn.Linear(128, num_classes)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 1st Convolutional Block
        x = self.conv1(x)
        x = self.relu1(x)
        
        # Pooling Layer
        x = self.pool1(x)
        
        # Flatten the output
        x = self.flatten(x)
        
        # Fully Connected Layer with ReLU
        x = self.fc1(x)
        x = self.relu2(x)
        
        # Output Layer
        x = self.fc2(x)
        return x
    

    
def load(dropout_p):
    new_num_classes = 4
    model = SimpleCNN(new_num_classes)
    if dropout_p is not None:
        model = dropout.add_dropout(model, "fc", dropout_p)
    return model

# experiment params:
# use_dropout: True, False
# weight_decay: 0.001, 0.0001, 0.00001, None (w/ use_dropout = False)
# learning_rate: 0.001, 0.0001, 0.00001
# epochs: 30
# patience: 5
def train_simplecnn(
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