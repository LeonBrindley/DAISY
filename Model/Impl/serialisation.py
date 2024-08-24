import os
import torch
import pickle

def save_val_preds(val_predictions, dir):
    os.makedirs(dir, exist_ok=True)

    # Save the validation predictions using pickle
    val_preds_path = os.path.join(dir, "val_predictions.pkl")
    with open(val_preds_path, 'wb') as f:
        pickle.dump(val_predictions, f)
    print(f"Saved validation predictions to {val_preds_path}")


def save_model(
    model, train_stats, val_stats, val_predictions, dir): # dir = "/rds/user/omsst2/hpc-work/gp/results/multiclass_plantnet"

    os.makedirs(dir, exist_ok=True)

    model_path = os.path.join(dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Save the training statistics using pickle
    train_stats_path = os.path.join(dir, "train_stats.pkl")
    with open(train_stats_path, 'wb') as f:
        pickle.dump(train_stats, f)
    print(f"Saved training statistics to {train_stats_path}")

    # Save the validation statistics using pickle
    val_stats_path = os.path.join(dir, "val_stats.pkl")
    with open(val_stats_path, 'wb') as f:
        pickle.dump(val_stats, f)
    print(f"Saved validation statistics to {val_stats_path}")
    
    save_val_preds(val_predictions, dir)


def load_model(dir, base_model):
    model_path = os.path.join(dir, "model.pth")
    train_stats_path = os.path.join(dir, "train_stats.pkl")
    val_stats_path = os.path.join(dir, "val_stats.pkl")
    val_preds_path = os.path.join(dir, "val_predictions.pkl")

    # Load the model state dictionary
    base_model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")
    
    # Load the training statistics
    with open(train_stats_path, 'rb') as f:
        train_stats = pickle.load(f)
    print(f"Loaded training statistics from {train_stats_path}")
    
    # Load the validation statistics
    with open(val_stats_path, 'rb') as f:
        val_stats = pickle.load(f)
    print(f"Loaded validation statistics from {val_stats_path}")

    # Load the validation predictions
    with open(val_preds_path, 'rb') as f:
        val_predictions = pickle.load(f)
    print(f"Loaded validation predictions from {val_preds_path}")
    
    return base_model, train_stats, val_stats, val_predictions

def load_only_model(dir, base_model):
    model_path = os.path.join(dir, "model.pth")
    # Load the model state dictionary
    base_model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")
    return base_model

def load_only_stats(dir):
    train_stats_path = os.path.join(dir, "train_stats.pkl")
    val_stats_path = os.path.join(dir, "val_stats.pkl")
    val_preds_path = os.path.join(dir, "val_predictions.pkl")

    # Load the training statistics
    with open(train_stats_path, 'rb') as f:
        train_stats = pickle.load(f)
    print(f"Loaded training statistics from {train_stats_path}")

    # Load the validation statistics
    with open(val_stats_path, 'rb') as f:
        val_stats = pickle.load(f)
    print(f"Loaded validation statistics from {val_stats_path}")

    # Load the validation predictions
    with open(val_preds_path, 'rb') as f:
        val_predictions = pickle.load(f)
    print(f"Loaded validation predictions from {val_preds_path}")

    return train_stats, val_stats, val_predictions


