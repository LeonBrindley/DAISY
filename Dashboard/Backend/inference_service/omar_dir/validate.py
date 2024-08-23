import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
import keras
import torch
from tqdm import tqdm

plt.ion()

def validation_inference(model, val_dataset):
    # Detect if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    val_predictions = []

    with torch.no_grad():
        for data in tqdm(val_dataset):
            inputs, labels = data
            inputs = inputs.unsqueeze(0).to(device)   # Add batch dimension and move to device

            output = model(inputs)
            predictions = (output > 0.5).float()      # Threshold value of 0.5

            val_predictions.append(predictions.cpu().detach().numpy())

    val_predictions = np.squeeze(np.array(val_predictions), axis=1)

    return val_predictions


def show_training_history(train_stats, val_stats, title):

  epochs = range(1, len(train_stats) + 1)
  train_losses = [entry[0] for entry in train_stats]
  train_accuracies = [entry[1] for entry in train_stats]
  val_losses = [entry[0] for entry in val_stats]
  val_accuracies = [entry[1] for entry in val_stats]

  fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))

  fig.suptitle(title)

  train_colour = 'tab:blue'
  val_colour = 'tab:red'
  # Plotting loss
  ax1.plot(epochs, train_losses, label='Train Loss', color=train_colour)
  ax1.plot(epochs, val_losses, label='Val Loss', color=val_colour)
  ax1.set_ylabel('Loss')
  ax1.legend()

  # Plot accuracy
  ax2.plot(epochs, train_accuracies, label='Train Accuracy', color=train_colour)
  ax2.plot(epochs, val_accuracies, label='Val Accuracy', color=val_colour)
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.legend()

  plt.tight_layout()
  return plt.show()


def show_accuracy(val_dataset, val_predictions):
    true_labels = []
    for _, labels in val_dataset:
        true_labels.append(labels.numpy())
    true_labels = np.vstack(true_labels)

    sklearn_accuracy = accuracy_score(true_labels, val_predictions)

    keras_bin_acc = keras.metrics.BinaryAccuracy()
    keras_bin_acc.update_state(true_labels, val_predictions)
    keras_acc = keras_bin_acc.result()
    keras_bin_acc.reset_state()

    print(f"Exact match (sklearn) accuracy: {sklearn_accuracy}")
    print(f"Keras BinaryAccuracy:           {keras_acc}")

    return sklearn_accuracy, keras_acc


def show_confusion(val_dataset, label_map, val_predictions):
    true_labels = []
    for _, labels in val_dataset:
        true_labels.append(labels.numpy())
    true_labels = np.vstack(true_labels)

    class_names = [label for label in label_map.keys()]

    
    confusion_matrices = multilabel_confusion_matrix(true_labels, val_predictions)

    num_classes = len(class_names)
    grid_size = int(np.ceil(np.sqrt(num_classes)))

    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(10, 10))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Plot each confusion matrix
    for i, (confusion_matrix, class_name) in enumerate(zip(confusion_matrices, class_names)):
        ax = axes[i]
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(class_name)

        # Show all ticks and label them with the respective list entries
        ax.set(xticks=np.arange(confusion_matrix.shape[1]),
            yticks=np.arange(confusion_matrix.shape[0]),
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = confusion_matrix.max() / 2.
        for j in range(confusion_matrix.shape[0]):
            for k in range(confusion_matrix.shape[1]):
                ax.text(k, j, format(confusion_matrix[j, k], fmt),
                        ha="center", va="center",
                        color="white" if confusion_matrix[j, k] > thresh else "black")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # Plot normalised confusion matrices

    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(15, 15))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Plot each confusion matrix
    for i, (confusion_matrix, class_name) in enumerate(zip(confusion_matrices, class_names)):
        ax = axes[i]
        
        # Normalize the confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

        # Calculate rates
        TP = confusion_matrix[1, 1]
        TN = confusion_matrix[0, 0]
        FP = confusion_matrix[0, 1]
        FN = confusion_matrix[1, 0]
        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0  # True Positive Rate
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0  # False Positive Rate
        TNR = TN / (TN + FP) if (TN + FP) != 0 else 0  # True Negative Rate
        FNR = FN / (FN + TP) if (FN + TP) != 0 else 0  # False Negative Rate
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{class_name}\nTPR: {TPR:.2f}, FPR: {FPR:.2f}, TNR: {TNR:.2f}, FNR: {FNR:.2f}")

        # Show all ticks and label them with the respective list entries
        ax.set(xticks=np.arange(confusion_matrix.shape[1]),
            yticks=np.arange(confusion_matrix.shape[0]),
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm_normalized.max() / 2.
        for j in range(cm_normalized.shape[0]):
            for k in range(cm_normalized.shape[1]):
                ax.text(k, j, format(cm_normalized[j, k], fmt),
                        ha="center", va="center",
                        color="white" if cm_normalized[j, k] > thresh else "black")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()