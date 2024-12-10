# Directory Structure and Main Components

## Directory Structure

| **Directory/File**       | **Description**                                                                 |
|---------------------------|---------------------------------------------------------------------------------|
| `project/`               | Contains main Python scripts for audio generation and processing.              |
| `generate.py`    | Script for generating MIDI, WAV, and CSV files with random musical patterns.    |
| `project/inference.py`   | Script for performing inference using a trained DETRAudio model.               |
| `project/train.py`       | Script for training the DETRAudio model using generated data.                  |
| `project/detraudio.py`   | Defines the DETRAudio model and related architectures.                         |
| `project/criterion.py`   | Defines custom criterion/loss functions for model training.                    |
| `project/dataset.py`     | Defines `AudioDataset` and data transformations.                               |
| `project/engine.py`      | Provides `train_one_epoch` and `evaluate` routines for training/evaluation loops. |
| `project/utils/`         | Utility scripts for data handling, logging, and other tasks.                   |
| `Fluid_related/`         | Contains SoundFont files (.sf2) used for converting MIDI to WAV.               |
| `output/`                | Directory where generated and predicted outputs (MIDI, WAV, CSV, metadata) are stored. |

---

# Main Components

### 1. **Data Generation (`generate.py`)**
- **Functionality**:  
  Generates a set of MIDI files with random notes, chords, and instruments.  
  Each MIDI file is converted to:  
  - A **WAV file** for audio representation.  
  - A **CSV file** listing note events.  

- **Purpose**:  
  This synthetic dataset is used to train the DETRAudio model. The data need to be moved and alight with train config file

---

### 2. **Dataset (`dataset.py`)**
- **Functionality**:  
  The dataset class (`AudioDataset`) loads generated audio/labels.  
  - Applies transformations like **Spectrogram**, **MelSpectrogram**, or **CQT**.  
  - Caches transformed data in a database.  
  - Prepares data for training or inference.

- **Label Files**:  
  Associated `.csv` files provide ground truth for:  
  - Note type  
  - Instrument  
  - Pitch  
  - Start time  
  - Duration  
  - Velocity  

---

### 3. **Model (`detraudio.py`)**
- **Overview**:  
  Defines the DETRAudio model, inspired by **DETR (Detection Transformers)**.  

- **Architecture**:  
  - **Backbone**: ResNet or custom CNN for feature extraction.  
  - **Transformer**: Encoder-decoder structure for sequence modeling.  
  - **Heads**:  
    - Classification heads: Note type, instrument, and pitch.  
    - Regression heads: Start time, duration, and velocity.  

---

### 4. **Training (`train.py`)**
- **Functionality**:  
  Trains the DETRAudio model using the generated dataset.  

- **Features**:  
  - Supports various schedulers and layer freezing options.  
  - Stores model checkpoints in the `model_save_path` directory.  
  - Logs training and validation metrics in CSV files.  

---

### 5. **Inference (`inference.py`)**
- **Functionality**:  
  Loads a trained DETRAudio model and runs inference on new audio.  

- **Outputs**:  
  - A **CSV** of predicted notes and instruments.  
  - Converts predictions back to a MIDI file.  
  - Optionally generates a WAV file using a SoundFont.  

# Instructions

## **Data Generation**
- Run `generate.py` to create synthetic datasets.  
- The script will produce a set of **MIDI**, **WAV**, and **CSV** files in the `output/` directory.

---

## **Model Training**
1. Prepare a `config.json` file with the training parameters.  
2. Run the following command:  
   ```bash
   python train.py --config config.json
   ```
# Inference Script Documentation

## **inference**
The `inference.py` script performs inference using a trained DETRAudio model on an input audio file, supporting chunk-based processing for large files. The script generates predictions, reconstructs them into MIDI and WAV formats, and aggregates metadata.

---

## **Usage**
### **Basic Command**
```bash
python inference.py --input path/to/input.wav --output_dir output_directory --model_path path/to/checkpoint.pth --config config.json --sound_font Fluid_related/FluidR3_GM.sf2
```

# Explanation of Configuration File and Model

## Configuration File (`config.json`)

The `config.json` file defines the parameters for training, data processing, model structure, and other settings. Here's a breakdown of its key components:

### 1. **General Settings**
| Parameter             | Description                                                                                 |
|-----------------------|---------------------------------------------------------------------------------------------|
| `version`             | Version identifier for the configuration.                                                   |
| `data_dir`            | Directory containing input data.                                                            |
| `cache_dir`           | Directory to store cached data.                                                             |
| `logs_dir`            | Directory to store logs during training.                                                    |
| `batch_size`          | Number of samples per training batch.                                                       |
| `num_workers`         | Number of workers for data loading.                                                         |
| `epochs`              | Number of training epochs.                                                                  |
| `debug`               | If `true`, enables debug mode for additional logging.                                       |

### 2. **Learning and Optimization**
| Parameter             | Description                                                                                 |
|-----------------------|---------------------------------------------------------------------------------------------|
| `lr`                  | Learning rate for the optimizer.                                                            |
| `scheduler_type`      | Learning rate scheduler type (`None`, `ExponentialLR`, `ReduceLROnPlateau`, etc.).          |
| `scheduler_params`    | Parameters for the learning rate scheduler.                                                 |
| `weight_decay`        | L2 regularization coefficient.                                                              |
| `gradient_accumulation_steps` | Number of gradient accumulation steps.                                              |

### 3. **Model Training and Saving**
| Parameter             | Description                                                                                 |
|-----------------------|---------------------------------------------------------------------------------------------|
| `model_save_path`     | Directory to save trained model checkpoints.                                                |
| `save_by_spoch`       | Frequency (in epochs) to save model checkpoints.                                            |

### 4. **Model Architecture**
| Parameter             | Description                                                                                 |
|-----------------------|---------------------------------------------------------------------------------------------|
| `model_structure`     | Settings for model architecture:                                                            |
| - `dimension`         | Base dimension of the model embeddings.                                                     |
| - `num_encoder_layers`| Number of transformer encoder layers.                                                       |
| - `num_decoder_layers`| Number of transformer decoder layers.                                                       |
| - `backbone_type`     | Type of backbone (`resnet18`, `resnet50`, `CustomTokenizedBackbone`, etc.).                  |
| - `positional_embedding` | Type of positional embedding (`sinusoid`, `2d`, `None`).                                 |
| - `time_series_type`  | Time series processing type (`default`, `LSTM`, `RNN`, etc.).                                |
| - `regression_activation_last_layer` | Activation function for the regression head.                                  |

### 5. **Dataset and Transforms**
| Parameter             | Description                                                                                 |
|-----------------------|---------------------------------------------------------------------------------------------|
| `transforms`          | Specifies the type and parameters of data transforms (e.g., `CQT`, `MelSpectrogram`).       |
| `max_objects`         | Maximum number of objects (notes) to detect in a sample.                                    |

### 6. **Loss and Costs**
| Parameter             | Description                                                                                 |
|-----------------------|---------------------------------------------------------------------------------------------|
| `loss`                | Settings for loss computation (e.g., `use_softmax`).                                        |
| `cost_*`              | Weights for different loss components (e.g., `cost_note_type`, `cost_pitch`).               |

---

## Model (`DETRAudio`)

The `DETRAudio` model is inspired by Detection Transformers (DETR) and adapted for audio data processing. It consists of the following components:

### 1. **Backbone**
- Extracts features from input spectrograms.
- Options include:
  - Pretrained ResNet (`resnet18`, `resnet50`).
  - Custom backbones like `TokenizedBackbone` or `CustomTokenizedBackbone`.
  - A simple CNN (`SimpleCNN`).

### 2. **Input Projection**
- Reduces the feature dimensionality to match the transformer input dimension using a `1x1` convolution.

### 3. **Positional Encoding**
- Adds positional information to the feature embeddings.
- Types:
  - `sinusoid`: Standard sinusoidal positional encoding.
  - `2d`: Two-dimensional encoding for spatial features.
  - `None`: No positional encoding.

### 4. **Transformer**
- Encoder-decoder architecture for sequence modeling.
- Supports customization of:
  - Number of encoder and decoder layers.
  - Integration with time-series models like RNN or LSTM.

### 5. **Query Embeddings**
- Learnable embeddings representing objects (notes) to be detected.

### 6. **Classification Heads**
- Predicts categorical attributes:
  - `note_type`: Type of musical note.
  - `instrument`: Instrument type.
  - `pitch`: Pitch value.

### 7. **Regression Heads**
- Predicts continuous attributes:
  - `start_time`: Note start time.
  - `duration`: Note duration.
  - `velocity`: Note intensity.

### 8. **Special Features**
- **Multiple Transformers**: Separate transformers for classification and regression tasks (enabled in `mv2` version).
- **Skip Connections**: Configurable multi-layer perceptron (MLP) with skip connections for better learning.

---

### Forward Pass Workflow
1. **Feature Extraction**:
   - Input spectrogram processed by the backbone.
   - Positional encoding is added.

2. **Sequence Modeling**:
   - Features passed through transformer encoders and decoders.
   - Separate processing for classification and regression in `mv2` version.

3. **Output Heads**:
   - Outputs are predicted through classification and regression heads.

4. **Final Output**:
   - Returns predictions for:
     - `note_type`
     - `instrument`
     - `pitch`
     - `start_time`
     - `duration`
     - `velocity`

---

This model, combined with the configurable settings in `config.json`, provides a flexible and robust framework for audio-based sequence detection and classification.



