import torch
import keras
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from torchmetrics.classification import MultilabelF1Score
import gc
from torch.cuda.amp import GradScaler, autocast


new_num_classes = 4


def fine_tune_model(model, device, train_loader, val_loader, optimiser, loss_function, num_epochs=20, patience=5):

    start = time.time()

    # PyTorch-native metrics to avoid CPU/GPU transfer overhead
    train_f1_metric = MultilabelF1Score(num_labels=new_num_classes).to(device)
    val_f1_metric = MultilabelF1Score(num_labels=new_num_classes).to(device)

    epoch_train_stats = []
    epoch_val_stats = []

    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_model_weights = None

    scaler = GradScaler()  # For mixed precision training

    def evaluate_model(model, data_loader, loss_function, f1_metric):
        model.eval()
        running_loss = 0.0
        num_batches = 0
        with torch.no_grad():
             for inputs, targets in tqdm(data_loader, leave=False):
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs)
                loss = loss_function(logits, targets)

                f1_metric.update(logits, targets)

                running_loss += loss.item()
                num_batches += 1

                del logits, loss

        epoch_loss = running_loss / num_batches
        epoch_f1 = f1_metric.compute().item()
        f1_metric.reset()

        return epoch_loss, epoch_f1

    initial_train_loss, initial_train_f1 = evaluate_model(model, train_loader, loss_function, train_f1_metric)
    epoch_train_stats.append((initial_train_loss, initial_train_f1))

    initial_val_loss, initial_val_f1 = evaluate_model(model, val_loader, loss_function, val_f1_metric)
    epoch_val_stats.append((initial_val_loss, initial_val_f1))

    print(f'Initial Train loss: {initial_train_loss:.2f}, Train f1: {initial_train_f1:.2f} '
                      f'Val loss: {initial_val_loss:.2f}, Val f1: {initial_val_f1:.2f}')

    torch.cuda.empty_cache()
    gc.collect()

    for epoch in range(num_epochs):

        print(f"epoch {epoch+1}/{num_epochs}")

        model.train()

        running_loss = 0.0
        num_batches = 0

        for inputs, targets in tqdm(train_loader, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimiser.zero_grad()

            with autocast():  # Mixed precision
                logits = model(inputs)
                loss = loss_function(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            train_f1_metric.update(logits, targets)

            running_loss += loss.item()
            num_batches += 1

            del logits, loss

        train_epoch_loss = running_loss / num_batches
        train_epoch_f1 = train_f1_metric.compute().item()
        epoch_train_stats.append((train_epoch_loss, train_epoch_f1))

        model.eval()
        val_running_loss = 0.0
        val_num_batches = 0

        with torch.no_grad():
                        for x_batch_val, y_batch_val in tqdm(val_loader, leave=False):
                            x_batch_val = x_batch_val.to(device)
                            y_batch_val = y_batch_val.to(device)

                            with autocast():  # Mixed precision
                                        val_logits = model(x_batch_val)
                                        val_loss = loss_function(val_logits, y_batch_val)

                            val_f1_metric.update(val_logits, y_batch_val)

                            val_running_loss += val_loss.item()
                            val_num_batches += 1

                            del val_logits, val_loss

        val_epoch_loss = val_running_loss / val_num_batches
        val_epoch_f1 = val_f1_metric.compute().item()
        
        epoch_val_stats.append((val_epoch_loss, val_epoch_f1))

        train_f1_metric.reset()
        val_f1_metric.reset()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train loss: {train_epoch_loss:.2f}, Train f1: {train_epoch_f1:.2f}, Val loss: {val_epoch_loss:.2f}, Val f1: {val_epoch_f1:.2f}')

        if val_epoch_f1 > best_val_f1:
            best_val_f1 = val_epoch_f1
            epochs_no_improve = 0
            best_model_weights = model.state_dict().copy()
            print(f"Best model saved with f1: {best_val_f1:.2f}")
        else:
            epochs_no_improve += 1
            print(f"Validation f1 not improved {epochs_no_improve} times.")

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

        torch.cuda.empty_cache()

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print("Loaded the best model weights.")

    elapsed_time = time.time() - start
    print('Fine-tuning complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))

    return model, epoch_train_stats, epoch_val_stats


def fine_tune_model_aux(model, device, train_loader, val_loader, optimiser, loss_function, num_epochs=20, patience=5):

    start = time.time()

    # PyTorch-native metrics to avoid CPU/GPU transfer overhead
    train_f1_metric = MultilabelF1Score(num_labels=new_num_classes).to(device)
    val_f1_metric = MultilabelF1Score(num_labels=new_num_classes).to(device)

    epoch_train_stats = []
    epoch_val_stats = []

    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_model_weights = None

    scaler = GradScaler()  # For mixed precision training

    def evaluate_model(model, data_loader, loss_function, f1_metric):
        model.eval()
        running_loss = 0.0
        num_batches = 0
        with torch.no_grad():
             for inputs, targets in tqdm(data_loader, leave=False):
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs)
                loss = loss_function(logits, targets)

                f1_metric.update(logits, targets)

                running_loss += loss.item()
                num_batches += 1

                del logits, loss

        epoch_loss = running_loss / num_batches
        epoch_f1 = f1_metric.compute().item()
        f1_metric.reset()

        return epoch_loss, epoch_f1

    initial_train_loss, initial_train_f1 = evaluate_model(model, train_loader, loss_function, train_f1_metric)
    epoch_train_stats.append((initial_train_loss, initial_train_f1))

    initial_val_loss, initial_val_f1 = evaluate_model(model, val_loader, loss_function, val_f1_metric)
    epoch_val_stats.append((initial_val_loss, initial_val_f1))

    print(f'Initial Train loss: {initial_train_loss:.2f}, Train f1: {initial_train_f1:.2f} '
                      f'Val loss: {initial_val_loss:.2f}, Val f1: {initial_val_f1:.2f}')

    torch.cuda.empty_cache()
    gc.collect()

    for epoch in range(num_epochs):

        print(f"epoch {epoch+1}/{num_epochs}")

        model.train()

        running_loss = 0.0
        num_batches = 0

        for inputs, targets in tqdm(train_loader, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimiser.zero_grad()

            with autocast():  # Mixed precision
                logits = model(inputs)

                if isinstance(logits, tuple):
                    logits, aux_logits = logits
                    loss1 = loss_function(logits, targets)
                    loss2 = loss_function(aux_logits, targets)
                    loss = loss1 + 0.4 * loss2
                else:
                    loss = loss_function(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            train_f1_metric.update(logits, targets)

            running_loss += loss.item()
            num_batches += 1

            del logits, loss

        train_epoch_loss = running_loss / num_batches
        train_epoch_f1 = train_f1_metric.compute().item()
        epoch_train_stats.append((train_epoch_loss, train_epoch_f1))

        model.eval()
        val_running_loss = 0.0
        val_num_batches = 0

        with torch.no_grad():
                        for x_batch_val, y_batch_val in tqdm(val_loader, leave=False):
                            x_batch_val = x_batch_val.to(device)
                            y_batch_val = y_batch_val.to(device)

                            with autocast():  # Mixed precision
                                        val_logits = model(x_batch_val)
                                        val_loss = loss_function(val_logits, y_batch_val)

                            val_f1_metric.update(val_logits, y_batch_val)

                            val_running_loss += val_loss.item()
                            val_num_batches += 1

                            del val_logits, val_loss

        val_epoch_loss = val_running_loss / val_num_batches
        val_epoch_f1 = val_f1_metric.compute().item()
        
        epoch_val_stats.append((val_epoch_loss, val_epoch_f1))

        train_f1_metric.reset()
        val_f1_metric.reset()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train loss: {train_epoch_loss:.2f}, Train f1: {train_epoch_f1:.2f}, Val loss: {val_epoch_loss:.2f}, Val f1: {val_epoch_f1:.2f}')

        if val_epoch_f1 > best_val_f1:
            best_val_f1 = val_epoch_f1
            epochs_no_improve = 0
            best_model_weights = model.state_dict().copy()
            print(f"Best model saved with f1: {best_val_f1:.2f}")
        else:
            epochs_no_improve += 1
            print(f"Validation f1 not improved {epochs_no_improve} times.")

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

        torch.cuda.empty_cache()

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print("Loaded the best model weights.")

    elapsed_time = time.time() - start
    print('Fine-tuning complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))

    return model, epoch_train_stats, epoch_val_stats


def freeze_weights(model):
      # freezes all weights
  for param in model.parameters():
            param.requires_grad = False


def unfreeze_weights(model, names):
        for name, param in model.named_parameters():
                    for unfreeze_name in names:
                                    if unfreeze_name in name or "fc1" in name or "fc2" in name:
                                                        param.requires_grad = True

def unfreeze_all_weights(model):
        for name, param in model.named_parameters():
                    param.requires_grad = True
