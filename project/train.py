# train.py
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from utils.dataset import AudioDataset
from models.detr_audio import DETRAudio
from utils.engine import train_one_epoch, evaluate
from utils.criterion import CustomCriterion
from config import CONFIG
from torchaudio.transforms import MelSpectrogram, Spectrogram
import os
import csv
from utils.utils import save_tensor_as_png , get_latest_checkpoint
def main():
    device = CONFIG['device']

    # Create logs directory if it doesn't exist
    logs_dir = CONFIG.get('logs_dir', 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Define paths for training and evaluation metrics
    train_metrics_path = os.path.join(logs_dir, 'train_metrics.csv')
    eval_metrics_path = os.path.join(logs_dir, 'eval_metrics.csv')

    # Initialize CSV files with headers if they don't exist
    if not os.path.exists(train_metrics_path):
        with open(train_metrics_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'average_loss', 'average_accuracy'])
        print(f"Created training metrics file at {train_metrics_path}")
    
    if not os.path.exists(eval_metrics_path):
        with open(eval_metrics_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'average_loss', 'average_accuracy'])
        print(f"Created evaluation metrics file at {eval_metrics_path}")

    # Data transforms
    transforms = Spectrogram(
        n_fft=4096,       # Double the FFT components for higher frequency resolution
        win_length=4096,  # Match win_length to n_fft for maximum resolution
        hop_length=256,   # Smaller hop length for higher time resolution
        power=2           # Use power spectrogram for amplitude squared
    )    

    # Datasets and DataLoaders
    print("Loading data from directory:", CONFIG['data_dir'])

    all_audio_files = [f for f in os.listdir(CONFIG['data_dir']) if f.endswith('.wav')]
    print(f"Found {len(all_audio_files)} audio files.")
    
    if len(all_audio_files) == 0:
        raise ValueError(f"No '.wav' files found in directory: {CONFIG['data_dir']}")

    # Optionally, print a sample file
    print(f"Sample audio file: {all_audio_files[0]}")

    # Train-validation split
    import random
    random.seed(CONFIG.get('random_seed', 42))  # For reproducibility
    random.shuffle(all_audio_files)
    split = int(0.8 * len(all_audio_files))
    train_audio_files = all_audio_files[:split]
    val_audio_files = all_audio_files[split:]
    print(f"Training samples: {len(train_audio_files)}, Validation samples: {len(val_audio_files)}")

    train_dataset = AudioDataset(
        train_audio_files, 
        CONFIG['cache_dir'], 
        split='train', 
        transforms=transforms, 
        config=CONFIG
    )
    val_dataset = AudioDataset(
        val_audio_files, 
        CONFIG['cache_dir'], 
        split='val', 
        transforms=transforms, 
        config=CONFIG
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'],
        collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        collate_fn=val_dataset.collate_fn
    )

    # Model, criterion, optimizer
    model = DETRAudio(CONFIG).to(device)
    criterion = CustomCriterion(CONFIG).to(device)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['lr'], 
        weight_decay=CONFIG['weight_decay']
    )

    # Optionally resume from a checkpoint
    start_epoch = 0
    version = CONFIG.get('version', 'v0')
    model_save_dir = CONFIG['model_save_path']
    latest_checkpoint, start_epoch = get_latest_checkpoint(model_save_dir, version)

    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {latest_checkpoint}, Epoch {checkpoint['epoch']}")
    else:
        start_epoch = 1  # Start from epoch 1 if no checkpoint is found
        print("No checkpoint found. Starting training from scratch.")

    def save_checkpoint(model_save_dir, version, epoch, model, optimizer, CONFIG):
        """
        Saves the model checkpoint with the specified version and epoch.

        Args:
            model_save_dir (str): Directory where model checkpoints are saved.
            version (str): Current version identifier.
            epoch (int): Current epoch number.
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            CONFIG (dict): Configuration dictionary.
        """
        checkpoint_path = os.path.join(model_save_dir, f"{version}_model_e{epoch}.pth")
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': CONFIG
            }, checkpoint_path)
            print(f"Saved checkpoint at {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    # Training Loop
    for epoch in range(start_epoch, CONFIG['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")

        # Train for one epoch
        avg_train_loss, avg_train_acc = train_one_epoch(
            model, 
            criterion, 
            train_loader, 
            optimizer, 
            device, 
            epoch + 1, 
            CONFIG
        )
        print(f"Training   - Epoch: {epoch + 1}, Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.4f}")
        try:
            with open(train_metrics_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, f"{avg_train_loss:.4f}", f"{avg_train_acc:.4f}"])
            print(f"Appended training metrics to {train_metrics_path}")
        except Exception as e:
            print(f"Error writing to training metrics file: {e}")

        if epoch % 5 == 0:
        # Evaluate on validation set
            avg_val_loss, avg_val_acc = evaluate(
                model, 
                criterion, 
                val_loader, 
                device
            )
            print(f"Validation - Epoch: {epoch + 1}, Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}")

            # Append training metrics to CSV


            # Append evaluation metrics to CSV
            try:
                with open(eval_metrics_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch + 1, f"{avg_val_loss:.4f}", f"{avg_val_acc:.4f}"])
                print(f"Appended evaluation metrics to {eval_metrics_path}")
            except Exception as e:
                print(f"Error writing to evaluation metrics file: {e}")

        # Save model checkpoint
    save_checkpoint(
        model_save_dir=model_save_dir,
        version=version,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        CONFIG=CONFIG
    )

    print("\nTraining complete.")

if __name__ == '__main__':
    main()
