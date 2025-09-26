"""
ViT to TimeSeries Model with Transformer Decoder and Cross-Attention

Combines custom rectangular ViT encoder with Transformer decoder
using CORAL domain bridging and proper cross-attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict, Any, Tuple
import numpy as np
import math
import sys
import os
from einops.layers.torch import Rearrange

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vit_encoder import RectangularViT, create_rectangular_vit
from pytorch_stft import get_STFT_spectra  # Importing the STFT function from pytorch_stft.py

import numpy as np
import matplotlib.pyplot as plt

class ValueEmbedding(nn.Module):
    def __init__(self, seq_channels, embed_channels):
        super(ValueEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=seq_channels, out_channels=embed_channels,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

'''
class DecoderPositionalEncoding(nn.Module):
    """
    Dynamic positional encoding for transformer decoder that computes encodings on-the-fly.
    Allows for variable prediction lengths without fixed buffer size limitations.
    Maintains backward compatibility with checkpoints trained using fixed PositionalEncoding.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        # Pre-compute div_term for efficiency (this doesn't depend on sequence length)
        self.register_buffer('div_term', torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        ))
        
        # Cache for computed positional encodings to avoid recomputation
        self._pe_cache = {}
    
    def _compute_pe(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute positional encoding for given sequence length."""
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
            
        # Check cache first
        cache_key = (seq_len, device.type, device.index if device.index is not None else 0)
        if cache_key in self._pe_cache:
            return self._pe_cache[cache_key]
        
        # Compute positional encoding
        pe = torch.zeros(seq_len, self.d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = self.div_term.to(device)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
        
        # Cache the result (limit cache size to prevent memory issues)
        if len(self._pe_cache) < 100:  # Reasonable cache limit
            self._pe_cache[cache_key] = pe
            
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        """
        seq_len = x.size(1)
        device = x.device
        
        # Get or compute positional encoding for this sequence length
        pe = self._compute_pe(seq_len, device)
        
        # Add positional encoding and apply dropout
        x = x + pe
        return self.dropout(x)
    
    def clear_cache(self):
        """Clear the positional encoding cache."""
        self._pe_cache.clear()
'''

class TransformerDecoderLayer(nn.Module):
    """
    Custom transformer decoder layer with self-attention and cross-attention.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target sequence from decoder (batch_size, tgt_len, d_model)
            memory: Encoder output (batch_size, src_len, d_model)
            tgt_mask: Causal mask for target sequence
            
        Returns:
            Output tensor (batch_size, tgt_len, d_model)
        """
        # Self-attention with residual connection
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with residual connection
        # Q from decoder (tgt), K and V from encoder (memory)
        tgt2 = self.cross_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward with residual connection
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerDecoderWithCrossAttention(nn.Module):
    """
    Transformer decoder with proper cross-attention mechanism.
    
    Q comes from decoder self-attention
    K, V come from encoder (ViT) output
    Additional conditioning from time series context
    """
    
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        prediction_length: int = 96,
        context_length: int = 96,
        time_series_dim: int = 1,
        encoder_dim: int = 512,  # ViT encoder output dimension
    ):
        super().__init__()
        
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.time_series_dim = time_series_dim
        
        # Embedding from time series values to model embedding
        self.last_seq_value_proj = nn.Linear(time_series_dim, d_model)
        
        # Dynamic positional encoding for variable prediction lengths
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Project encoder output to decoder dimension for cross-attention
        self.encoder_projection = nn.Linear(encoder_dim, d_model)

        # Project context condition (last feature of input series) [b, seq_len] to decoder dimension [b, d_mode] for cross-attention
        self.context_condition_projection = nn.Linear(context_length, d_model)
        
        # Custom transformer decoder layers with cross-attention
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, time_series_dim)
        )
        
        # Initialize parameters with better initialization schemes
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """
        Initialize decoder parameters with better schemes for gradient sensitivity.
        """
        
        # Initialize value embedding layer
        nn.init.xavier_uniform_(self.last_seq_value_proj.weight)
        nn.init.constant_(self.last_seq_value_proj.bias, 0.0)
        
        # Initialize encoder projection
        nn.init.xavier_uniform_(self.encoder_projection.weight)
        nn.init.constant_(self.encoder_projection.bias, 0.0)

        
        # Initialize transformer decoder layers with scaled initialization
        for layer in self.decoder_layers:
            # Self-attention
            self._init_multihead_attention(layer.self_attn)
            # Cross-attention  
            self._init_multihead_attention(layer.cross_attn)
            # Feed-forward layers
            self._init_feedforward(layer.ffn)
            # Layer norms
            nn.init.constant_(layer.norm1.weight, 1.0)
            nn.init.constant_(layer.norm1.bias, 0.0)
            nn.init.constant_(layer.norm2.weight, 1.0)
            nn.init.constant_(layer.norm2.bias, 0.0)
            nn.init.constant_(layer.norm3.weight, 1.0)
            nn.init.constant_(layer.norm3.bias, 0.0)
        
        # Initialize output projection with smaller weights for stable training
        for i, layer in enumerate(self.output_projection):
            if isinstance(layer, nn.Linear):
                if i == len(self.output_projection) - 1:  # Final layer
                    # Smaller initialization for final output layer
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
                else:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
    
    def _init_multihead_attention(self, attention_layer):
        """Initialize multihead attention layer with proper scaling."""
        # Access the underlying linear layers
        # MultiheadAttention has in_proj_weight (for Q, K, V) and out_proj
        if hasattr(attention_layer, 'in_proj_weight') and attention_layer.in_proj_weight is not None:
            # Combined Q, K, V projection
            nn.init.xavier_uniform_(attention_layer.in_proj_weight)
        else:
            # Separate Q, K, V projections
            if hasattr(attention_layer, 'q_proj_weight'):
                nn.init.xavier_uniform_(attention_layer.q_proj_weight)
            if hasattr(attention_layer, 'k_proj_weight'):
                nn.init.xavier_uniform_(attention_layer.k_proj_weight) 
            if hasattr(attention_layer, 'v_proj_weight'):
                nn.init.xavier_uniform_(attention_layer.v_proj_weight)
        
        # Initialize biases
        if hasattr(attention_layer, 'in_proj_bias') and attention_layer.in_proj_bias is not None:
            nn.init.constant_(attention_layer.in_proj_bias, 0.0)
        
        # Output projection
        nn.init.xavier_uniform_(attention_layer.out_proj.weight, gain=1/math.sqrt(2))  # Residual scaling
        nn.init.constant_(attention_layer.out_proj.bias, 0.0)
    
    def _init_feedforward(self, ffn_module):
        """Initialize feed-forward network with residual scaling."""
        for i, layer in enumerate(ffn_module):
            if isinstance(layer, nn.Linear):
                if i == len(ffn_module) - 1:  # Final layer in FFN
                    # Scale down final layer for residual connections
                    nn.init.xavier_uniform_(layer.weight, gain=1/math.sqrt(2))
                else:
                    nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self, 
        condition: torch.Tensor,
        encoder_output: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing.
        
        Args:
            decoder_input: Last feature column of context (batch_size, context_length)
            encoder_output: Output from ViT encoder (batch_size, num_patches+1, encoder_dim)
            target: Ground truth for teacher forcing (batch_size, prediction_length, time_series_dim)
            use_teacher_forcing: Whether to use teacher forcing (True during training)
            
        Returns:
            Predictions (batch_size, prediction_length, time_series_dim)
        """
        batch_size = condition.size(0)
        pred_len = self.prediction_length
        device = condition.device
        
        if use_teacher_forcing:
            # Teacher forcing: use ground truth as input, properly aligned for prediction
            # Input: [start_token, target[0], target[1], ..., target[n-2]]
            # Output: [target[0], target[1], target[2], ..., target[n-1]]

            # Project teacher forcing target to the same space of decoder_input embedding  
            decoder_input = target[:, :, -self.time_series_dim:]  # (batch_size, pred_len, time_series_dim)
            decoder_input = self.last_seq_value_proj(decoder_input)  # (batch_size, pred_len, d_model)
            
            # Use the last value of the context as the start token (also using same projection)
            last_context_value = condition[:, -1:, -self.time_series_dim:]
            start_token = self.last_seq_value_proj(last_context_value)

            decoder_input = torch.cat([start_token, decoder_input[:,:-1,:]], dim=1)  # (batch_size, pred_len, ts_dim)
            decoder_input = self.pos_encoding(decoder_input)

            # Create causal mask
            tgt_len = decoder_input.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)
            
            # Pass through decoder layers
            output = decoder_input
            for layer in self.decoder_layers:
                #print(f"output shape: {output.shape}, encoder_output shape: {encoder_output.shape}")
                #print(f"encoder_output shape: {encoder_output.shape}")
                output = layer(output, encoder_output, tgt_mask=tgt_mask)
                
            
            # Project to output dimension - now directly predicts target
            output = self.output_projection(output)  # (batch_size, pred_len, ts_dim)
            
            # Output directly corresponds to target without shifting
            
        else:
            # Inference mode: autoregressive generation
            predictions = []

            # Use the last value of the context as the start token
            last_context_value = condition[:, -1:, -self.time_series_dim:]
            start_token = self.last_seq_value_proj(last_context_value)

            # Pre-allocate sequence tensor (will be filled progressively)
            batch_size = start_token.size(0)
            d_model = start_token.size(2)
            device = start_token.device

            # Initialize with start token, rest will be filled during generation
            sequence_embeddings = torch.zeros(batch_size, self.prediction_length, d_model, device=device)
            sequence_embeddings[:, 0:1, :] = start_token

            for step in range(self.prediction_length):
                # Get current sequence (only filled portion)
                current_length = step + 1
                current_input = sequence_embeddings[:, :current_length, :]

                # Apply PE to current sequence (automatic based on sequence length)
                current_input = self.pos_encoding(current_input)

                # Create causal mask
                tgt_mask = self._generate_square_subsequent_mask(current_length, device)

                # Pass through decoder layers
                output = current_input
                for layer in self.decoder_layers:
                    output = layer(output, encoder_output, tgt_mask=tgt_mask)

                # Get prediction for next time step
                next_pred_value = self.output_projection(output[:, -1:, :])
                predictions.append(next_pred_value)

                # If not the last step, prepare next token embedding
                if step < self.prediction_length - 1:
                    sequence_embeddings[:, step+1:step+2, :] = self.last_seq_value_proj(next_pred_value)
            output = torch.cat(predictions, dim=1)


        return output


class ViTToTimeSeriesModel(nn.Module):
    """
    Architecture combining rectangular ViT encoder with Transformer decoder.
    
    Uses linear projection for feature adaptation and cross-attention mechanism.
    Supports teacher forcing during training and autoregressive generation during inference.
    """
    
    def __init__(
        self,
        num_channels: int = 1,
        prediction_length: int = 96,
        context_length: int = 96,
        feature_projection_dim: int = 128,
        time_series_dim: int = 1,
        ts_model_dim: int = 64,
        ts_num_heads: int = 8,
        ts_num_layers: int = 4,
        ts_dim_feedforward: int = 1024,
        ts_dropout: float = 0.1,
    ):
        """
        Initialize the model.
        
        Args:
            num_channels: Number of channels in spectrogram
            prediction_length: Length of time series to predict
            context_length: Length of context window
            feature_projection_dim: Dimension for feature projection
            time_series_dim: Dimension of time series (usually 1 for univariate)
            ts_model_dim: Hidden dimension for transformer decoder
            ts_num_heads: Number of attention heads
            ts_num_layers: Number of decoder layers
            ts_dim_feedforward: Feed-forward dimension
            ts_dropout: Dropout rate
            image_mean: Mean values for image normalization
            image_std: Std values for image normalization
        """
        super().__init__()
        
        # Store configuration
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.time_series_dim = time_series_dim
        self.feature_projection_dim = feature_projection_dim
        self.ts_model_dim = ts_model_dim
        self.num_channels = num_channels
        self.embed_dim = 512
        
        # Value embedding for time series input with too many channels
        if num_channels > 32:
            self.embed_channels_dim = 32  # Embedding dimension for each channel of ValueEmbedding
        else:
            self.embed_channels_dim = 16  # Embedding dimension for each channel of ValueEmbedding
        self.value_embedding = ValueEmbedding(seq_channels=num_channels, embed_channels=self.embed_channels_dim)

        # Rectangular ViT Encoder (64*64 spectrograms)
        
        '''
        self.vit_encoder = create_rectangular_vit(
            image_height=64,  # Updated height for resized spectrograms
            image_width=64,  # Updated width for resized spectrograms
            in_channels=self.embed_channels_dim,
            embed_dim=768,
            depth=1,
            num_heads=8,
            mlp_ratio=4,
            dropout=0.1
        )
        '''

        #self.encoder_projection = nn.Linear(vit_hidden_size, feature_projection_dim)
        
        
        #self.my_resnet.conv1 = nn.Conv2d(self.embed_channels_dim, 64, kernel_size=8, stride=8, padding=0, bias=False) # 64 comes from self.inplanes of resnet lib
        #self.my_resnet.fc = nn.Linear(512, vit_hidden_size)
        # conv1: 64×64 → 8×8
        self.my_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT,progress=False)
        self.my_resnet.conv1 = nn.Conv2d(self.embed_channels_dim, 64, kernel_size=8, stride=8, padding=0, bias=False)

        self.my_resnet.maxpool = nn.Sequential(
            # 特征提取
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 8×8 → 8×8
            
            # 渐进上采样：8×8 → 28×28 → 56×56
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=3, padding=1, output_padding=1),  # ≈28×28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 28×28 → 56×56
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            
            # 最后用dilated conv细化特征
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.my_resnet.avgpool = nn.Identity()  # 去掉avgpool
        self.my_resnet.fc = Rearrange('b (h w c) -> b (h w) c', h=6, w=6, c=512)

        # 修改maxpool为stride=1，保持8×8的特征提取能力
        #self.my_resnet.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 保持8×8

        # 在layer1后截断，此时特征图是8×8
        # layer1: 8×8 → 8×8 (stride=1)
        #self.my_resnet.layer2 = nn.Identity()
        #self.my_resnet.layer3 = nn.Identity()  
        #self.my_resnet.layer4 = nn.Identity()
        #self.my_resnet.avgpool = nn.Identity()
        #self.my_resnet.fc = nn.Linear(64*8*8, self.embed_dim)  # 输入维度调整为64*8*8
        
        # Transformer Decoder with Cross-Attention
        self.ts_decoder = TransformerDecoderWithCrossAttention(
            d_model=ts_model_dim,
            nhead=ts_num_heads,
            num_layers=ts_num_layers,
            dim_feedforward=ts_dim_feedforward,
            dropout=ts_dropout,
            prediction_length=prediction_length,
            context_length=context_length,
            time_series_dim=time_series_dim,
            encoder_dim=feature_projection_dim,  # After linear projection
        )
        
        # self.my_resnet = MultiScaleResNet(embed_channels_dim=self.embed_channels_dim, d_model=cnn_hidden_size)
        # TSLib standard: no additional normalization in model (handled in data loader)
    
    def forward(self, context: torch.Tensor, tf_target: torch.Tensor = None, mode: str = 'train') -> torch.Tensor:
        """
        Forward pass with TSLib standard preprocessing (normalized input from data loader).
        
        Args:
            context: Input context time series (batch_size, context_length, num_features) - already StandardScaler normalized
            tf_target: Target time series for teacher forcing (batch_size, prediction_length, num_features) - already normalized
            mode: 'train' for teacher forcing, 'inference' for autoregressive generation
            
        Returns:
            Predicted time series (batch_size, prediction_length, 1) - normalized scale (TSLib standard)
        """
        device = next(self.parameters()).device
        batch_size = context.size(0)

        context = self.value_embedding(context)
        
        # Step 1: Get last feature column of context as condition (already normalized by StandardScaler)
        condition = context[:, :, -self.time_series_dim:]  # (batch, context_len, time_series_dim)
        
        # Step 2: Generate spectrograms from normalized context
        spectra_list = []
        for item in context:
            spectra = get_STFT_spectra(item, device=device)
            spectra_list.append(spectra)
        
        # Stack into batch tensor
        spectra_tensor = torch.stack(spectra_list, dim=0)  # (batch, channels, img_height, img_width)

        #print(f"spectra_tensor shape: {spectra_tensor.shape}")
        cnn_features = self.my_resnet(spectra_tensor)   # (batch, vit_hidden_size)
        #print(f"cnn_features shape: {cnn_features.shape}")
        #print(f"cnn_features shape: {cnn_features.shape}")
        #cnn_features = cnn_features.unsqueeze(1)    # (batch, 1, vit_hidden_size), 1 is the one representing the whole image's encoded feature
        #print(f"cnn_features shape: {cnn_features.shape}")

        #encoder_features = self.encoder_projection(cnn_features)  # (batch, num_patches+1, feature_projection_dim)
        # Step 3: Process through ViT encoder
        #vit_features = self.vit_encoder.get_last_hidden_state(spectra_tensor)  # (batch, num_patches+1, 768)
        #encoder_features = self.encoder_projection(vit_features)  # (batch, num_patches+1, feature_projection_dim)
        
        # Step 4: Decoder forward pass
        if mode == 'train':            
            predictions = self.ts_decoder(
                condition=condition,
                encoder_output=cnn_features,  # Pass context condition for cross-attention
                target=tf_target,
                use_teacher_forcing=True
            )
        else:
            # Inference: autoregressive generation
            predictions = self.ts_decoder(
                condition=condition,
                encoder_output=cnn_features,  # Pass context condition for cross-attention
                target=None,
                use_teacher_forcing=False
            )
        
        # Return normalized predictions (TSLib standard for loss computation)
        return predictions
    
    def inference(self, context: torch.Tensor) -> torch.Tensor:
        """
        Inference mode without teacher forcing.
        
        Args:
            context: Input context time series (batch_size, context_length, features)
            
        Returns:
            Predicted time series (batch_size, prediction_length, time_series_dim)
        """
        return self.forward(context=context, tf_target=None, mode='inference')
    
    #def freeze_vit_encoder(self, freeze: bool = True):
    #    """Freeze or unfreeze the ViT encoder parameters."""
    #    for param in self.vit_encoder.parameters():
    #        param.requires_grad = not freeze
    
    def freeze_ts_decoder(self, freeze: bool = True):
        """Freeze or unfreeze the Transformer decoder parameters."""
        for param in self.ts_decoder.parameters():
            param.requires_grad = not freeze


def create_model(
    prediction_length: int = 96,
    context_length: int = 96,
    **kwargs
) -> ViTToTimeSeriesModel:
    """
    Factory function to create ViTToTimeSeriesModel with common configurations.
    
    Args:
        prediction_length: Length of predictions
        context_length: Length of context window
        **kwargs: Additional model arguments
        
    Returns:
        Initialized model
    """
    return ViTToTimeSeriesModel(
        prediction_length=prediction_length,
        context_length=context_length,
        **kwargs
    )