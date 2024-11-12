import torch
import torch.nn as nn
import math
from torchvision.models import resnet50 ,ResNet50_Weights
import math
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights

class DETRAudio(nn.Module):
    def __init__(self, config):
        super(DETRAudio, self).__init__()
        dimension = config["model_structure"].get("dimension", 128)
        num_encoder_layers = config["model_structure"].get("num_encoder_layers", 1)
        num_decoder_layers = config["model_structure"].get("num_decoder_layers", 1)
        self.num_queries = config['max_objects']
        self.dimension = dimension
        self.debug = config.get('debug', False)
        if self.debug:
            print("----->","Debug mode enabled")

        # Backbone selection
        backbone_type = config["model_structure"].get('backbone_type', 'Resnet')
        pretrain = config.get('pretrain', False)
        if backbone_type == 'None':
            # Simple CNN to match the transformer output size
            print("----->","Using Simple CNN backbone")
            self.backbone = SimpleCNN()
            backbone_output_dim = self.backbone.output_dim
        elif backbone_type == 'resnet18':
            # Use ResNet backbone
            print("-----> Using ResNet-18 backbone with pretrain:", pretrain)
            if pretrain:
                self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = resnet18(weights=False)
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            backbone_output_dim = 512
        elif backbone_type == 'resnet50':
            # Use ResNet backbone
            print("----->","Using ResNet-50 backbone with pretrain:", pretrain)
            if pretrain:
                self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_NO_TOP)
            else:
                self.backbone = resnet50(weights=False)
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            backbone_output_dim = 2048
        elif backbone_type == 'TokenizedBackbone':
            # Other custom backbone
            print("----->","Using TokenizedBackbone")
            self.backbone = TokenizedBackbone(embed_dim=dimension)
            backbone_output_dim = self.backbone.output_dim
        elif backbone_type == 'CustomTokenizedBackbone':
            # Custom Tokenized Backbone
            print("----->","Using CustomTokenizedBackbone")
            self.backbone = CustomTokenizedBackbone(embed_dim=dimension,CONFIG=config)
            backbone_output_dim = self.backbone.output_dim

        # Input projection to match transformer dimension
        self.input_proj = nn.Conv2d(backbone_output_dim, dimension, kernel_size=1)

        # Positional embedding
        positional_embedding = config["model_structure"].get('positional_embedding', 'None')
        if positional_embedding == 'sinusoid':
            # 1D sin-cos positional encoding
            print("----->","Using sinusoidal positional encoding")
            self.positional_encoding = PositionalEncoding(dimension)
        elif positional_embedding == '2d':
            # 2D ViT-like positional encoding
            print("----->","Using 2D positional encoding")
            self.positional_encoding = TwoDPositionalEncoding(dimension)
        else:
            print("----->","No positional encoding")
            self.positional_encoding = None  # No positional encoding

        # Time series model selection
        time_series_type = config["model_structure"].get('time_series_type', 'default')
        if time_series_type == 'default':
            print("----->","Using default transformer")
            self.transformer = nn.Transformer(d_model=dimension, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        elif time_series_type == 'SparseFormer':
            print("----->","Using SparseFormer")
            self.transformer = SparseFormer(d_model=dimension, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        elif time_series_type == 'RNN':
            print("----->","Using RNN")
            self.transformer = StackedRNN(d_model=dimension, num_layers=num_encoder_layers)
        elif time_series_type == 'LSTM':
            print("----->","Using LSTM")
            self.transformer = StackedLSTM(d_model=dimension, num_layers=num_encoder_layers)
        else:
            raise ValueError(f"Unknown time_series_type: {time_series_type}")

        # Query embeddings for transformer decoder
        self.query_embed = nn.Embedding(self.num_queries, dimension)

        # Classification and regression heads
        number_of_layers = config["model_structure"].get('number_of_layers', 2)
        activation_last_layer = 'sigmoid'  # As per default setting
        num_classes_note_type = config['num_classes']['note_type'] + 1
        num_classes_instrument = config['num_classes']['instrument'] + 1
        num_classes_pitch = config['num_classes']['pitch'] + 1

        # Classification heads with configurable number of layers
        self.class_embed_note_type = MLP(dimension, dimension, num_classes_note_type, number_of_layers, activation_last_layer)
        self.class_embed_instrument = MLP(dimension, dimension, num_classes_instrument, number_of_layers, activation_last_layer)
        self.class_embed_pitch = MLP(dimension, dimension, num_classes_pitch, number_of_layers, activation_last_layer)

        # Regression head with specified activation function
        regression_activation_last_layer = config["model_structure"].get('regression_activation_last_layer', 'relu')
        print("----->","Regression activation:", regression_activation_last_layer)
        #self.bbox_embed = MLP(dimension, dimension, 3, number_of_layers, regression_activation_last_layer,config.get("r_head_scaler",10))  # For start_time, duration, velocity
        regression_activation_last_layer = config["model_structure"].get('regression_activation_last_layer', 'relu')

        self.start_time_head = MLP(
            dimension, dimension, 1, number_of_layers, regression_activation_last_layer, config.get("start_time_scaler", 20)
        )
        self.duration_head = MLP(
            dimension, dimension, 1, number_of_layers, regression_activation_last_layer, config.get("duration_scaler", 2)
        )
        self.velocity_head = MLP(
            dimension, dimension, 1, number_of_layers, regression_activation_last_layer, config.get("velocity_scaler", 200)
        )
        print("----->","START TIME SCALER:", config.get("start_time_scaler", 20))
        print("----->","DURATION SCALER:", config.get("duration_scaler", 2))
        print("----->","VELOCITY SCALER:", config.get("velocity_scaler", 200))
        print("----->","DETRAudio model initialized")

    def forward(self, x):
        # x: [batch_size, 1, freq_bins, time_steps]
        bs = x.size(0)
        if self.debug:
            print("----->","Input shape:", x.shape)
        x = self.backbone_conv(x)  # Backbone processing
        if self.debug:
            print("----->","Backbone output shape:", x.shape)
        x = self.input_proj(x)     # [batch_size, dimension, H, W]
        if self.debug:
            print("----->","Input projection shape:", x.shape)
        # Flatten spatial dimensions and apply positional encoding if any
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        if self.debug:
            print("----->","Positional encoding shape:", x.shape)
        x = x.flatten(2).permute(2, 0, 1)  # [seq_len, batch_size, dimension]
        if self.debug:
            print("----->","Flattened shape:", x.shape)
        # Time series model processing
        if isinstance(self.transformer, nn.Transformer):
            memory = self.transformer.encoder(x)
            hs = self.transformer.decoder(self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1), memory)
            if self.debug:
                print("----->","Transformer output shape:", hs.shape)
        else:
            hs = self.transformer(x)
            hs = hs[-self.num_queries:]  # Select the last 'num_queries' outputs
            if self.debug:
                print("----->","Time series model output shape:", hs.shape)
        hs = hs.permute(1, 0, 2)  # [batch_size, num_queries, dimension]

        # Classification and regression heads
        outputs_note_type = self.class_embed_note_type(hs)
        outputs_instrument = self.class_embed_instrument(hs)
        outputs_pitch = self.class_embed_pitch(hs)
        # Regression outputs
        outputs_start_time = self.start_time_head(hs)
        outputs_duration = self.duration_head(hs)
        outputs_velocity = self.velocity_head(hs)

        outputs_regression = torch.cat([outputs_start_time, outputs_duration, outputs_velocity], dim=-1)


        self.debug = False
        return {
            'pred_note_type': outputs_note_type,
            'pred_instrument': outputs_instrument,
            'pred_pitch': outputs_pitch,
            'pred_regression': outputs_regression
        }

    def backbone_conv(self, x):
        if hasattr(self.backbone, 'conv1'):
            # ResNet backbone
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
        else:
            # Custom backbone
            x = self.backbone(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Simple CNN matching transformer input dimension
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Input is spectrogram
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.output_dim = 256  # Adjust as needed

    def forward(self, x):
        return self.conv_layers(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenizedBackbone(nn.Module):
    def __init__(
        self,
        input_channels=1,
        embed_dim=256,
        patch_size=(16, 16),
        stride=(8, 8),
        padding=(0, 0),
        num_conv_layers=2,
        dropout=0.1
    ):
        """
        Tokenizes spectrogram input into overlapping patches suitable for transformers.

        Args:
            input_channels (int): Number of input channels (1 for grayscale spectrograms).
            embed_dim (int): Dimension of the token embeddings.
            patch_size (tuple): Size of each patch (height, width).
            stride (tuple): Stride for convolution (controls overlap).
            padding (tuple): Padding for convolution.
            num_conv_layers (int): Number of convolutional layers to process patches.
            dropout (float): Dropout rate for regularization.
        """
        super(TokenizedBackbone, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = embed_dim

        # Initial convolution to create patch embeddings
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )  # Output: (B, embed_dim, H_patch, W_patch)

        # Optional additional convolutional layers for deeper feature extraction
        conv_layers = []
        for _ in range(num_conv_layers):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            conv_layers.append(nn.BatchNorm2d(embed_dim))
            conv_layers.append(nn.ReLU(inplace=True))
        self.additional_convs = nn.Sequential(*conv_layers) if num_conv_layers > 0 else nn.Identity()

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms of shape (B, C, H, W).

        Returns:
            torch.Tensor: Token embeddings of shape (B, N_patches, embed_dim).
        """
        B, C, H, W = x.shape

        # Apply initial convolution to extract patches
        x = self.conv(x)  # Shape: (B, embed_dim, H_patch, W_patch)
        x = self.additional_convs(x)  # Shape: (B, embed_dim, H_patch, W_patch)

        # # Flatten spatial dimensions to create tokens
        # x = x.flatten(2)  # Shape: (B, embed_dim, N_patches)
        # x = x.transpose(1, 2)  # Shape: (B, N_patches, embed_dim)

        # Apply layer normalization and dropout
        # x = self.layer_norm(x)
        # x = self.dropout(x)

        return x








import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the model's embeddings.
            max_len (int, optional): The maximum sequence length. Defaults to 50000.
        """
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape [max_len, d_model] with positional encodings
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)  # Register as buffer to avoid being considered as a model parameter
        self.dimension = d_model

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, d_model, H, W] or [batch_size, channel, d_model, H, W].

        Returns:
            torch.Tensor: Tensor with positional encoding added, maintaining the original shape.
        """
        original_shape = x.shape
        num_dims = x.dim()

        if num_dims == 4:
            # Input shape: [batch_size, d_model, H, W]
            batch_size, d_model, H, W = x.size()
            # Flatten spatial dimensions
            seq_len = H * W
            x = x.view(batch_size, d_model, seq_len)  # [batch_size, d_model, H*W]
            # Permute to [H*W, batch_size, d_model] for adding positional encoding
            x = x.permute(2, 0, 1)  # [H*W, batch_size, d_model]
            # Add positional encoding
            if seq_len > self.pe.size(0):
                raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.pe.size(0)}")
            x = x + self.pe[:seq_len, :]  # Broadcasting to [H*W, batch_size, d_model]
            # Permute back and reshape to original spatial dimensions
            x = x.permute(1, 2, 0)  # [batch_size, d_model, H*W]
            x = x.view(batch_size, d_model, H, W)  # [batch_size, d_model, H, W]

        elif num_dims == 5:
            ## TODO: Implement positional encoding for 5D input
            # Input shape: [batch_size, channel, d_model, H, W]
            batch_size, channel, d_model, H, W = x.size()
            seq_len = H * W
            # Reshape to combine batch and channel dimensions for processing
            x = x.view(batch_size * channel, d_model, seq_len)  # [batch_size*channel, d_model, H*W]
            # Permute to [H*W, batch_size*channel, d_model]
            x = x.permute(2, 0, 1)  # [H*W, batch_size*channel, d_model]
            # Add positional encoding
            if seq_len > self.pe.size(0):
                raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.pe.size(0)}")
            x = x + self.pe[:seq_len, :]  # Broadcasting to [H*W, batch_size*channel, d_model]
            # Permute back and reshape to original spatial dimensions
            x = x.permute(1, 2, 0)  # [batch_size*channel, d_model, H*W]
            x = x.view(batch_size, channel, d_model, H, W)  # [batch_size, channel, d_model, H, W]

        else:
            raise ValueError(f"Unsupported input shape {x.shape}. Expected 4 or 5 dimensions.")

        return x


import torch
import torch.nn as nn
import math

class TwoDPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_height=100, max_width=2000):
        """
        Initializes the TwoDPositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the model's embeddings. Must be even.
            max_height (int, optional): The maximum height of the input feature maps. Defaults to 64.
            max_width (int, optional): The maximum width of the input feature maps. Defaults to 64.
        """
        super(TwoDPositionalEncoding, self).__init__()
        
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for TwoDPositionalEncoding.")

        self.d_model = d_model
        self.max_height = max_height
        self.max_width = max_width

        # Initialize positional encoding tensor
        pe = torch.zeros(d_model, max_height, max_width)  # [d_model, max_height, max_width]
        
        # Create position indices for height and width
        y_position = torch.arange(0, max_height, dtype=torch.float).unsqueeze(1)  # [max_height, 1]
        x_position = torch.arange(0, max_width, dtype=torch.float).unsqueeze(0)   # [1, max_width]

        # Compute the div_term for sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]

        # Apply sine to even indices in the d_model dimension
        pe[0::2, :, :] = torch.sin(y_position * div_term.unsqueeze(1).unsqueeze(1))  # [d_model/2, max_height, max_width]

        # Apply cosine to odd indices in the d_model dimension
        pe[1::2, :, :] = torch.cos(x_position * div_term.unsqueeze(1).unsqueeze(1))  # [d_model/2, max_height, max_width]

        # Register pe as a buffer to ensure it's saved and moved with the model, but not updated during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds 2D positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, d_model, H, W] 
                              or [batch_size, channel, d_model, H, W].

        Returns:
            torch.Tensor: Tensor with positional encoding added, maintaining the original shape.
        """
        original_shape = x.shape
        num_dims = x.dim()

        if num_dims == 4:
            # Case 1: Input shape [batch_size, d_model, H, W]
            batch_size, d_model, H, W = x.size()

            if d_model != self.d_model:
                raise ValueError(f"Input d_model ({d_model}) does not match initialized d_model ({self.d_model}).")

            if H > self.max_height or W > self.max_width:
                raise ValueError(f"Input height ({H}) or width ({W}) exceeds maximum ({self.max_height}, {self.max_width}).")

            # Add positional encoding
            # pe has shape [d_model, max_height, max_width]
            # We slice pe to match the input height and width
            # Then broadcast to [batch_size, d_model, H, W]
            x = x + self.pe[:, :H, :W].unsqueeze(0)

        elif num_dims == 5:
            # Case 2: Input shape [batch_size, channel, d_model, H, W]
            batch_size, channel, d_model, H, W = x.size()

            if d_model != self.d_model:
                raise ValueError(f"Input d_model ({d_model}) does not match initialized d_model ({self.d_model}).")

            if H > self.max_height or W > self.max_width:
                raise ValueError(f"Input height ({H}) or width ({W}) exceeds maximum ({self.max_height}, {self.max_width}).")

            # Add positional encoding
            # pe has shape [d_model, max_height, max_width]
            # We slice pe to match the input height and width
            # Then broadcast to [batch_size, channel, d_model, H, W]
            x = x + self.pe[:, :H, :W].unsqueeze(0).unsqueeze(1)

        else:
            raise ValueError(f"Unsupported input shape {x.shape}. Expected 4 or 5 dimensions.")

        return x


class StackedRNN(nn.Module):
    def __init__(self, d_model, num_layers):
        super(StackedRNN, self).__init__()
        self.rnn = nn.RNN(input_size=d_model, hidden_size=d_model, num_layers=num_layers)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        output, _ = self.rnn(x)
        return output

class StackedLSTM(nn.Module):
    def __init__(self, d_model, num_layers):
        super(StackedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=num_layers)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        output, _ = self.lstm(x)
        return output

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron with configurable last layer activation"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation_last_layer=None,scaler=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        self.scaler = scaler
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        if not(activation_last_layer == "None"):
            if activation_last_layer.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation_last_layer.lower() == 'gelu':
                layers.append(nn.GELU())
            elif activation_last_layer.lower() == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation_last_layer.lower() == 'tanh':
                layers.append(nn.Tanh())
                layers.append(AddOne())
            else:
                raise ValueError(f"Unknown activation function: {activation_last_layer}")
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)*self.scaler
class AddOne(nn.Module):
    def __init__(self):
        super(AddOne, self).__init__()

    def forward(self, x):
        return x + 1
    
class SparseFormer(nn.Module):
    def __init__(self, d_model, num_encoder_layers, num_decoder_layers):
        super(SparseFormer, self).__init__()
        # Placeholder for SparseFormer implementation
        # TODO: Implement SparseFormer
        import warnings
        warnings.warn("SparseFormer is not implemented yet. Using default transformer.")
        # raise NotImplementedError("SparseFormer is not implemented yet. Using default transformer.")
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8), num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=8), num_layers=num_decoder_layers)

    def forward(self, x):
        memory = self.encoder(x)
        output = self.decoder(x, memory)
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomTokenizedBackbone(nn.Module):
    def __init__(
        self,
        input_channels=1,
        embed_dim=1024,  # Desired embedding dimension
        time_width=1,
        freq_bins=2049,  # Number of frequency bins (H)
        dropout=0.1,
        CONFIG=None
    ):
        """
        Custom Tokenized Backbone for DETRAudio Model.

        Args:
            input_channels (int): Number of input channels in the spectrogram.
            embed_dim (int): Dimension of the output token embeddings.
            time_width (int): Width of each time window for tokenization.
            freq_bins (int): Number of frequency bins in the spectrogram (H).
            dropout (float): Dropout rate for regularization.
        """
        super(CustomTokenizedBackbone, self).__init__()
        self.time_width = time_width
        self.freq_bins = freq_bins
        self.embed_dim = embed_dim

        # Step 1: Collapse channels using a learnable weighted mean
        self.collapse_channels = nn.Conv2d(
            in_channels=input_channels,
            out_channels=1,
            kernel_size=1,
            bias=False  # No bias to perform a weighted mean
        )

        # Step 2: Define a convolution layer that maintains the original shape
        self.patch_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,  # To maintain spatial dimensions
            bias=False
        )

        # Step 3: Adaptive pooling to map spatial dimensions to embed_dim
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, embed_dim))

        # Step 4: Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Final output dimension
        self.output_dim = embed_dim

    def forward(self, x):
        """
        Forward pass of the CustomTokenizedBackbone.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Token embeddings of shape (B, embed_dim, 1, W_patches).
        """
        B, C, H, W = x.shape
        # print(f"Input shape: {x.shape}")

        # Step 1: Collapse channels to 1 using a learnable weighted mean
        x = self.collapse_channels(x)  # Shape: (B, 1, H, W)
        # print("After collapse_channels:", x.shape)

        # Step 2: Tokenize the width dimension by windowing
        # Ensure W is divisible by time_width; if not, pad the width dimension
        if W % self.time_width != 0:
            pad_size = self.time_width - (W % self.time_width)
            x = F.pad(x, (0, pad_size))  # Pad the width dimension on the right
            W += pad_size  # Update W after padding
            # print(f"Padded width to {W}")

        W_patches = W // self.time_width  # Number of time windows
        # print(f"W_patches: {W_patches}")

        # Use unfold to create non-overlapping windows along the width dimension
        # After unfolding: (B, 1, H, W_patches, time_width)
        x = x.unfold(dimension=3, size=self.time_width, step=self.time_width)
        # print("After unfold:", x.shape)

        # Rearrange dimensions to (B, W_patches, 1, H, time_width)
        x = x.permute(0, 3, 1, 2, 4)
        # print("After permute:", x.shape)

        # Reshape to (B * W_patches, 1, H, time_width) for independent convolution
        x = x.contiguous().view(B * W_patches, 1, H, self.time_width)
        # print("After view (B * W_patches, 1, H, time_width):", x.shape)

        # Step 3: Apply independent convolution on each patch
        x = self.patch_conv(x)  # Shape: (B * W_patches, 1, H, time_width)
        # print("After patch_conv:", x.shape)

        # Step 4: Apply adaptive pooling to map spatial dimensions to embed_dim
        x = self.adaptive_pool(x)  # Shape: (B * W_patches, 1, 1, embed_dim)
        # print("After adaptive_pool:", x.shape)

        # Reshape to (B, W_patches, embed_dim)
        x = x.view(B, W_patches, self.output_dim)
        # print("After reshaping to (B, W_patches, embed_dim):", x.shape)

        # Step 5: Apply dropout
        x = self.dropout(x)
        # print("After dropout:", x.shape)

        # Step 6: Reshape to (B, embed_dim, 1, W_patches) to match other model dimensions
        x = x.permute(0, 2, 1).unsqueeze(2)  # Shape: (B, embed_dim, 1, W_patches)
        # print("Final output shape:", x.shape)

        return x





