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
from utils.utils import save_tensor_as_png, get_latest_checkpoint
import json

def main():
    device = CONFIG['device']
    version = CONFIG.get('version', 'v0')

    # Create logs directory if it doesn't exist
    logs_dir = CONFIG.get('logs_dir', 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Define paths for training and evaluation metrics
    train_metrics_path = os.path.join(logs_dir, f'{version}_train_metrics.csv')
    eval_metrics_path = os.path.join(logs_dir, f'{version}_eval_metrics.csv')
    confg_load_path = os.path.join(logs_dir, f'{version}_config.json')
    if not os.path.exists(confg_load_path):
        with open(confg_load_path, 'w') as f:
            json.dump(CONFIG, f, indent=4)
        print(f"Created config file at {confg_load_path}")

    # Define the metric keys based on your latest changes
    metric_keys = ['pit_acc', 'InstAcc', 'R_M_ST', 'R_M_dur', 'R_M_v']

    # Initialize CSV files with headers if they don't exist
    if not os.path.exists(train_metrics_path):
        with open(train_metrics_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Updated header to include all metric keys
            header = ['epoch', 'average_loss'] + metric_keys + ["batch_failed"]
            writer.writerow(header)
        print(f"Created training metrics file at {train_metrics_path}")

    if not os.path.exists(eval_metrics_path):
        with open(eval_metrics_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Updated header to include all metric keys
            header = ['epoch', 'average_loss'] + metric_keys + ["batch_failed"]
            writer.writerow(header)
        print(f"Created evaluation metrics file at {eval_metrics_path}")

    transforms_config = {
        "n_fft": CONFIG.get("data_nfft", 4096),
        "win_length": CONFIG.get("data_win_length", 4096),
        "hop_length": CONFIG.get("data_hop_length", 256),
        "power": CONFIG.get("data_power", 2)
    }

    # Example usage
    transforms = Spectrogram(**transforms_config)

    # Datasets and DataLoaders
    print("Loading data from directory:", CONFIG['data_dir'])

    all_audio_files = [f for f in os.listdir(CONFIG['data_dir']) if f.endswith('.wav')]

    # Initialize train and validation lists
    train_audio_files = []
    val_audio_files = []
    start_epoch = 0

    model_save_dir = CONFIG['model_save_path']
    latest_checkpoint, start_epoch = get_latest_checkpoint(model_save_dir, version)
    model = DETRAudio(CONFIG).to(device)
    model_config_path = os.path.join(logs_dir, f'{version}_model_config.txt')  # Changed extension for clarity
    if not os.path.exists(model_config_path):
        with open(model_config_path, 'w') as f:
            f.write(str(model))
    criterion = CustomCriterion(CONFIG).to(device)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['lr'], 
        weight_decay=CONFIG['weight_decay']
    )

    # Optionally resume from a checkpoint
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {latest_checkpoint}, Epoch {checkpoint['epoch']}")

        # Load the split from the checkpoint
        train_audio_files = checkpoint.get('train_audio_files', [])
        val_audio_files = checkpoint.get('val_audio_files', [])
        if not train_audio_files or not val_audio_files:
            raise ValueError("Checkpoint does not contain train and validation split information.")
    else:
        start_epoch = 1  # Start from epoch 1 if no checkpoint is found
        print("No checkpoint found. Starting training from scratch.")
        import random
        random.shuffle(all_audio_files)
        split_ratio = CONFIG.get('split_ratio', 0.8)
        split_index = int(len(all_audio_files) * split_ratio)
        train_audio_files = all_audio_files[:split_index]
        val_audio_files = all_audio_files[split_index:]

    train_dataset = AudioDataset(
        train_audio_files, 
        CONFIG['cache_dir'], 
        split='train', 
        transforms=transforms, 
        config=CONFIG,
        transforms_spec=transforms_config
    )
    val_dataset = AudioDataset(
        val_audio_files, 
        CONFIG['cache_dir'], 
        split='val', 
        transforms=transforms, 
        config=CONFIG,
        transforms_spec=transforms_config
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

    def save_checkpoint(model_save_dir, version, epoch, model, optimizer, CONFIG, train_audio_files, val_audio_files):
        """
        Saves the model checkpoint with the specified version and epoch, including train and validation splits.

        Args:
            model_save_dir (str): Directory where model checkpoints are saved.
            version (str): Current version identifier.
            epoch (int): Current epoch number.
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            CONFIG (dict): Configuration dictionary.
            train_audio_files (list): List of training audio filenames.
            val_audio_files (list): List of validation audio filenames.
        """
        checkpoint_path = os.path.join(model_save_dir, f"{version}_model_e{epoch}.pth")
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': CONFIG,
                'train_audio_files': train_audio_files,
                'val_audio_files': val_audio_files
            }, checkpoint_path)
            print(f"Saved checkpoint at {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    # Training Loop
    for epoch in range(start_epoch, CONFIG['epochs']):
        print(f"\n--- Epoch {epoch}/{CONFIG['epochs']} ---")

        # Train for one epoch
        avg_train_loss, aggregated_debug_metrics, batch_failed = train_one_epoch(
            model, 
            criterion, 
            train_loader, 
            optimizer, 
            device, 
            epoch, 
            CONFIG
        )
        print(f"Training   - Epoch: {epoch}, Loss: {avg_train_loss:.4f}")
        for key, value in aggregated_debug_metrics.items():
            print(f"            - {key}: {value}")

        # Extract metrics safely with default values
        try:
            pit_acc = aggregated_debug_metrics.get('pit_acc', 0.0)
            InstAcc = aggregated_debug_metrics.get('InstAcc', 0.0)
            R_M_ST = aggregated_debug_metrics.get('R_M_ST', 0.0)
            R_M_dur = aggregated_debug_metrics.get('R_M_dur', 0.0)
            R_M_v = aggregated_debug_metrics.get('R_M_v', 0.0)
        except KeyError as e:
            print(f"Missing metric in training: {e}")
            pit_acc = InstAcc = R_M_ST = R_M_dur = R_M_v = 0.0

        # Write training metrics to CSV
        try:
            with open(train_metrics_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch, 
                    f"{avg_train_loss:.4f}", 
                    f"{pit_acc:.4f}", 
                    f"{InstAcc:.4f}", 
                    f"{R_M_ST:.4f}", 
                    f"{R_M_dur:.4f}", 
                    f"{R_M_v:.4f}",
                    batch_failed
                ])
            print(f"Appended training metrics to {train_metrics_path}")
        except Exception as e:
            print(f"Error writing to training metrics file: {e}")

        # Evaluate on validation set every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_val_loss, aggregated_val_metrics, val_batch_failed = evaluate(
                model, 
                criterion, 
                val_loader, 
                device
            )
            print(f"Validation - Epoch: {epoch}, Loss: {avg_val_loss:.4f}")
            for key, value in aggregated_val_metrics.items():
                print(f"            - {key}: {value}")

            # Extract validation metrics safely with default values
            try:
                pit_acc_val = aggregated_val_metrics.get('pit_acc', 0.0)
                InstAcc_val = aggregated_val_metrics.get('InstAcc', 0.0)
                R_M_ST_val = aggregated_val_metrics.get('R_M_ST', 0.0)
                R_M_dur_val = aggregated_val_metrics.get('R_M_dur', 0.0)
                R_M_v_val = aggregated_val_metrics.get('R_M_v', 0.0)
            except KeyError as e:
                print(f"Missing metric in validation: {e}")
                pit_acc_val = InstAcc_val = R_M_ST_val = R_M_dur_val = R_M_v_val = 0.0

            # Write validation metrics to CSV
            try:
                with open(eval_metrics_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        epoch, 
                        f"{avg_val_loss:.4f}", 
                        f"{pit_acc_val:.4f}", 
                        f"{InstAcc_val:.4f}", 
                        f"{R_M_ST_val:.4f}", 
                        f"{R_M_dur_val:.4f}",
                        f"{R_M_v_val:.4f}",
                        val_batch_failed
                    ])
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
                CONFIG=CONFIG,
                train_audio_files=train_audio_files,
                val_audio_files=val_audio_files
            )

    print("\nTraining complete.")

if __name__ == '__main__':
    main()
