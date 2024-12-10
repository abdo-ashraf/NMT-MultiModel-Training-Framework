import torch
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import os

## Helper
@torch.no_grad()
def loss_acc_loader(model, data_loader, criterion, device):
    model.eval()

    total_predicts = []
    total_targets = []
    total_class_loss = 0

    for data, labels in data_loader:
        data = data.to(device)
        labels = labels.to(device)

        class_logits = model(data, labels)
        class_logits_flat = class_logits[:, 1:, :].reshape(-1, class_logits.size(-1))
        labels_flat = labels[:, 1:].reshape(-1)

        total_class_loss += criterion(class_logits_flat, labels_flat).item()

        predictions = class_logits_flat.argmax(-1)
        non_pad_mask = labels_flat != 0
        filtered_predictions = predictions[non_pad_mask]
        filtered_labels = labels_flat[non_pad_mask]

        total_predicts.append(filtered_predictions)
        total_targets.append(filtered_labels)

    total_predicts = torch.concat(total_predicts)
    total_targets = torch.concat(total_targets)

    avg_class_loss = total_class_loss / len(data_loader)
    accuracy = (total_predicts == total_targets).float().mean().item()

    return avg_class_loss, accuracy, total_predicts, total_targets


def get_parameters_info(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
    nontrainable = sum(p.numel() for p in model.parameters() if p.requires_grad==False)

    return trainable, nontrainable


def training(model, criterion, optimizer, train_loader, valid_loader, epochs, device):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5,
                                                           patience=5, min_lr=1e-6, threshold=0.001)
    total_batches = len(train_loader)
    train_class_losses = []
    val_class_losses = []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_losses = []

        tqdm_loop = tqdm(enumerate(train_loader), total=total_batches, position=0)
        for batch_idx, (data, labels) in tqdm_loop:
            # Get data
            data = data.to(device)
            labels = labels.to(device)
            # Forward
            class_logits = model(data, labels)

            class_logits_flat = class_logits[:, 1:, :].reshape(-1, class_logits.size(-1))
            labels_flat = labels[:, 1:].reshape(-1)
            loss = criterion(class_logits_flat, labels_flat)

            epoch_losses.append(loss.item())
            mean_epoch_loss = sum(epoch_losses)/len(epoch_losses)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            # Update progress bar
            tqdm_loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            tqdm_loop.set_postfix_str(f'loss = {round(mean_epoch_loss, 4)}')

        train_class_losses.append(mean_epoch_loss)

        val_class_loss, val_accuracy, _, _ = loss_acc_loader(model, valid_loader, criterion, device)
        val_class_losses.append(val_class_loss)
        print(f'Validation: Class Loss {val_class_loss:.4f}, Accuracy {val_accuracy*100:.2f}%')

        # Get the current learning rate from the optimizer
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step the scheduler with the validation loss
        scheduler.step(val_class_loss)
        
        # Check if the learning rate has changed
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f"Epoch {epoch + 1}: Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}")

    return train_class_losses, val_class_losses


def plot_loss(train_class_losses, val_class_losses, plots_dir, model_name):

    fig = plt.figure(figsize=(8, 3))
    epochs = [ep + 1 for ep in range(len(train_class_losses))]
    plt.plot(epochs, train_class_losses, label="Training Classification Loss")
    plt.plot(epochs, val_class_losses, label="Validation Classification Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()  # Add a legend to differentiate the lines
    plot_path = os.path.join(plots_dir, f'{model_name}_losses.png')
    plt.savefig(plot_path, dpi=300)
    # Close the plot to prevent it from displaying
    plt.close(fig)