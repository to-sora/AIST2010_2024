# config.py
import torch

CONFIG = {
    'version': 'v1',
    'data_dir': './data_2/',
    'cache_dir': './data/cache/',
    'logs_dir': './logs/',
    
    'batch_size': 2,
    'num_workers': 2,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'gradient_accumulation_steps': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_save_path': './models/model_data',
    'num_classes': {
        'note_type': 2,        # Single, Chord
        'instrument': 5,     # MIDI instrument numbers
        'pitch': 128            # Piano key range
    },
    'model_structure': {
        "dimension": 128,
        "num_encoder_layers": 1,
        "num_decoder_layers": 2,

    },
    'max_objects': 100 ,       # Maximum notes per audio sample
    "cost_note_type": 1,
    "cost_instrument": 1,
    "cost_pitch": 5,
    "cost_regression": 1,
    "data_nftt":4096,
    "data_win_length":4096,
    "data_hop_length":256,
    "data_power":2
}
