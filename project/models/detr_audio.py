# models/detr_audio.py
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn.functional import interpolate

class DETRAudio(nn.Module):
    def __init__(self, config):
        super(DETRAudio, self).__init__()
        dimension = config["model_structure"].get("dimension",128) 
        num_encoder_layers = config["model_structure"].get("num_encoder_layers",1)
        num_decoder_layers = config["model_structure"].get("num_decoder_layers",1)
        self.backbone = resnet50()
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.transformer = nn.Transformer(d_model=dimension, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        self.num_queries = config['max_objects']
        self.query_embed = nn.Embedding(self.num_queries, dimension)
        self.input_proj = nn.Conv2d(2048, dimension, kernel_size=1)

        # Heads for classification and regression
        self.class_embed_note_type = nn.Linear(dimension, config['num_classes']['note_type']+1)
        self.class_embed_instrument = nn.Linear(dimension, config['num_classes']['instrument']+1)
        self.class_embed_pitch = nn.Linear(dimension, config['num_classes']['pitch']+1)
        self.bbox_embed = MLP(dimension, dimension, 3, 3)  # For start_time, duration, velocity

    def forward(self, x):
        # x: [batch_size, 1, freq_bins, time_steps]
        bs = x.size(0)
        # print("start")
        # print(x.shape)
        x = self.backbone_conv(x)
        # print("backbone")
        # print(x.shape)
        x = self.input_proj(x)
        # print("input_proj")
        # print(x.shape)
        x = x.flatten(2).permute(2, 0, 1) # [time_steps * freq_bins, batch_size, 256]
        
        # print("ready tranformer ") 
        # print(x.shape)
        memory = self.transformer.encoder(x)
        # print(memory.shape)
        hs = self.transformer.decoder(self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1), memory)
        # print(hs.shape)
        # print("end transformer reshaping")
        hs = hs.permute(1, 0, 2)
        # print(hs.shape)

        # different mlp heads
        outputs_note_type = self.class_embed_note_type(hs)
        outputs_instrument = self.class_embed_instrument(hs)
        outputs_pitch = self.class_embed_pitch(hs)
        outputs_regression = self.bbox_embed(hs)

        return {
            'pred_note_type': outputs_note_type,
            'pred_instrument': outputs_instrument,
            'pred_pitch': outputs_pitch,
            'pred_regression': outputs_regression
        }

    def backbone_conv(self, x):
        # Apply backbone up to layer4
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
