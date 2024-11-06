# config.py
import torch

CONFIG = {
    'version': 'v1',
    'data_dir': './data_2/',
    'cache_dir': './data/cache/',
    'logs_dir': './logs/',
    
    'batch_size': 2,
    'num_workers': 1,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'gradient_accumulation_steps': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_save_path': './models/',
    'num_classes': {
        'note_type': 2,        # Single, Chord
        'instrument': 5,     # MIDI instrument numbers
        'pitch': 128            # Piano key range
    },
    'max_objects': 100 ,       # Maximum notes per audio sample
    "dimension": 64,
    "cost_note_type": 1,
    "cost_instrument": 1,
    "cost_pitch": 1,
    "cost_regression": 1
}
