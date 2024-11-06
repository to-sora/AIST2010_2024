# # utils/engine.py
import torch
from tqdm import tqdm

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, config):
    """
    Trains the model for one epoch and returns the average loss and accuracy.

    Args:
        model (torch.nn.Module): The model to train.
        criterion (callable): The loss function.
        data_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        config (dict): Configuration dictionary containing training parameters.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()
    # If the criterion has a train mode, enable it. Otherwise, this line can be removed.
    if hasattr(criterion, 'train'):
        criterion.train()
    optimizer.zero_grad()

    accum_steps = config.get('gradient_accumulation_steps', 1)
    scaler = torch.cuda.amp.GradScaler()
    loop = tqdm(data_loader, total=len(data_loader), desc=f"Epoch [{epoch}]")

    total_loss = 0.0
    total_accuracy = 0.0
    valid_steps = 0

    for i, data_dict in enumerate(loop):
        samples, targets = data_dict

        samples = samples.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        shape = samples.shape  # For error logging

        try:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss_dict, accuracy = criterion(outputs, targets)
                loss = sum(loss_dict.values()) / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(data_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Accumulate loss and accuracy
            total_loss += loss.item() * accum_steps  # Multiply back to get actual loss
            total_accuracy += accuracy.item()
            valid_steps += 1

            # Update tqdm postfix
            avg_loss = total_loss / valid_steps
            avg_accuracy = total_accuracy / valid_steps
            loop.set_postfix(loss=avg_loss, accuracy=avg_accuracy)

        except Exception as e:
            print(f"Error in training: {e}")
            print(f"SAMPLES shape: {shape}")

    # Compute final averages
    avg_loss = total_loss / valid_steps if valid_steps > 0 else 0.0
    avg_accuracy = total_accuracy / valid_steps if valid_steps > 0 else 0.0

    return avg_loss, avg_accuracy

def evaluate(model, criterion, data_loader, device):
    """
    Evaluates the model on the validation/test set and returns the average loss and accuracy.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (callable): The loss function.
        data_loader (DataLoader): DataLoader for validation/test data.
        device (torch.device): Device to run the evaluation on.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    # If the criterion has an eval mode, enable it. Otherwise, this line can be removed.
    if hasattr(criterion, 'eval'):
        criterion.eval()

    losses = []
    accuracies = []

    with torch.no_grad():
        loop = tqdm(data_loader, total=len(data_loader), desc="Evaluating")
        for samples, targets in loop:
            samples = samples.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            shape = samples.shape  # For error logging

            try:
                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                    loss_dict, accuracy = criterion(outputs, targets)
                    loss = sum(loss_dict.values())

                losses.append(loss.item())
                accuracies.append(accuracy.item())

                avg_loss = sum(losses) / len(losses)
                avg_accuracy = sum(accuracies) / len(accuracies)

                loop.set_postfix(loss=avg_loss, accuracy=avg_accuracy)

            except Exception as e:
                print(f"Error in evaluation: {e}")
                print(f"SAMPLES shape: {shape}")

    avg_loss = sum(losses) / len(losses) if len(losses) > 0 else 0.0
    avg_accuracy = sum(accuracies) / len(accuracies) if len(accuracies) > 0 else 0.0
    return avg_loss, avg_accuracy
