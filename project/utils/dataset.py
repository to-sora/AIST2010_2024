# utils/dataset.py
import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from typing import Callable, Optional, Dict, Any
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
class AudioDataset(Dataset):
    def __init__(self, audiofile, cache_dir, split='train', transforms=None,freq_limit=4200 ,config=None,transforms_spec=None):
        self.cache_dir = cache_dir
        self.freq_limit = freq_limit
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_prefix = ""
        if transforms is None:
            if transforms_spec is None:
                self.transforms = torchaudio.Spectrogram(n_fft=4096, win_length=4096, hop_length=256, power=2)
                cache_prefix += "4096_4096_256_2"
            else:
                self.transforms = transforms_spec
                for key in transforms_spec.keys():
                    cache_prefix += f"{key}_{transforms_spec[key]}_"
        else:
            self.transforms = transforms
            for key in transforms_spec.keys():
                cache_prefix += f"{key}_{transforms_spec[key]}_"
        # Get list of audio files
        self.audio_files = audiofile

        # debug
        self.audio_files = self.audio_files[:100]

        # Initialize lists to hold data in RAM
        self.spectrograms = []
        self.targets = []
        self.data_dir = config['data_dir']
        # Load all data into RAM
        for audio_filename in tqdm(self.audio_files, desc=f"Loading {split} data"):
            audio_path = os.path.join(self.data_dir, audio_filename)
            cache_path = os.path.join(self.cache_dir, cache_prefix + '_'+ audio_filename.replace('.wav', '.pt'))

            def load_audio(audio_path,cach_path):
                waveform, sample_rate = torchaudio.load(audio_path)
                if sample_rate != 44100:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, 44100)
                spectrogram = self.transforms(waveform)
                freq_res = 44100 / self.transforms.n_fft  # Frequency resolution per bin
                max_bin = int(self.freq_limit / freq_res)
                spectrogram = spectrogram[:max_bin, :].mean(dim=0).unsqueeze(0)
                # # print(spectrogram.shape)
                # assert False
                torch.save(spectrogram, cache_path)
                return spectrogram

            # Load or compute spectrogram
            if os.path.exists(cache_path):
                try:
                    spectrogram = torch.load(cache_path,weights_only=True)
                except:
                    spectrogram = load_audio(audio_path,cache_path)
            else:
                spectrogram = load_audio(audio_path,cache_path)

            # Load labels
            csv_filename = audio_filename.replace('.wav', '.csv')
            csv_path = os.path.join(self.data_dir, csv_filename)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Label file {csv_filename} not found for audio {audio_filename}")
            df = pd.read_csv(csv_path)

            # Process labels
            target = self.process_labels(df)

            # Store in RAM
            self.spectrograms.append(spectrogram)
            self.targets.append(target)
        # # print one sample to check
        # print("First sample")
        # print(self.spectrograms[0].shape)
        # print(self.targets[0])
        # # print last sample to check
        # print("Last sample")
        # print(self.spectrograms[-1].shape)
        # print(self.targets[-1])

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spectrogram = self.spectrograms[idx]
        target = self.targets[idx]
        return spectrogram, target

    def process_labels(self, df):
        # Convert labels to tensor format suitable for the model
        labels = {}

        # Map Note_Type to indices
        note_type_mapping = {'Single': 1, 'Chord': 2}
        labels['note_type'] = torch.tensor(df['Note_Type'].map(note_type_mapping).values, dtype=torch.long)

        # Map Instrument to indices (you may need to expand this mapping)
        labels['instrument'] = torch.tensor(df['Instrument'].apply(self.instrument_to_index).values, dtype=torch.long)

        # Adjust Pitch (MIDI note numbers)
        labels['pitch'] = torch.tensor(df['Pitch'].values - 21, dtype=torch.long)  # Adjusted for piano keys starting from A0 (MIDI 21)

        # Regression targets
        labels['start_time'] = torch.tensor(df['Start_Time'].values, dtype=torch.float32)
        labels['duration'] = torch.tensor(df['Duration'].values, dtype=torch.float32)
        labels['velocity'] = torch.tensor(df['Velocity'].values, dtype=torch.float32)

        return labels

    def instrument_to_index(self, instrument_name):
        # Map instrument names to indices (assuming mapping is predefined)
        instruments = [
            'Acoustic Grand Piano',
            'Violin',
            'Flute',
            'Electric Guitar (jazz)'
            # 'Synth Lead',
            # Add more instruments as needed
        ]
        instrument_dict = {instrument: idx + 1 for idx, instrument in enumerate(instruments)}
        # You may need to expand this mapping based on your dataset
        return instrument_dict.get(instrument_name, 0)  # Default to 0 if not found
    
    @staticmethod
    def collate_fn(batch: list) -> Dict[str, Any]:
        """
        Custom collate function to handle batches with variable-length spectrograms and labels.

        Args:
            batch (list): List of tuples (spectrogram, target).

        Returns:
            dict: Batched spectrograms and labels.
        """
        spectrograms, targets = zip(*batch)  # Unzip the batch

        # Find the maximum time dimension
        max_time = max([s.size(2) for s in spectrograms])

        # Initialize batch tensor
        batch_size = len(spectrograms)
        channels = spectrograms[0].size(0)
        freq_bins = spectrograms[0].size(1)

        # Create a tensor of zeros to hold the batch
        padded_spectrograms = torch.zeros(batch_size, channels, freq_bins, max_time)

        # Copy spectrograms into the batch tensor
        for i, s in enumerate(spectrograms):
            time = s.size(2)
            padded_spectrograms[i, :, :, :time] = s

        # Handle labels
        # Since labels are dictionaries with tensors, we'll need to collate each field separately
        collated_labels = {}
        label_keys = targets[0].keys()
        for key in label_keys:
            # Collect all tensors for this key
            label_list = [t[key] for t in targets]
            # print(key)


            if label_list[0].dim() == 1:
                # Variable-length sequences, pad them
                collated_labels[key] = pad_sequence(label_list, batch_first=True, padding_value=0)
            else:
                # Fixed-size tensors, stack them
                collated_labels[key] = torch.stack(label_list)
            # print(collated_labels[key].shape)        
        return [ padded_spectrograms,  collated_labels]

