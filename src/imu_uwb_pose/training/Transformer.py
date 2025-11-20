import torch
import torch.nn as nn
import pytorch_lightning as pl

class PoseTransformer(pl.LightningModule):
    def __init__(self, 
                 input_dim=15, 
                 output_dim=132, 
                 d_model=64, 
                 n_heads=4, 
                 n_layers=2, 
                 dropout=0.1):
        super().__init__()
        # Linear projection to embed input features into d_model dimensions
        self.input_proj = nn.Linear(input_dim, d_model)
        # Learnable positional embeddings for sequence length 150
        self.pos_embedding = nn.Parameter(torch.zeros(1, 150, d_model))
        # Transformer encoder: stack of n_layers self-attention blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                  dim_feedforward=d_model*4, 
                                                  dropout=dropout,
                                                  batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Output projection to desired output dimensions (22 joints * 6D = 132)
        self.output_proj = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the model.
        Args:
            x: Tensor of shape (B, T, input_dim), here T=150.
        Returns:
            Tensor of shape (B, T, output_dim) with the predicted pose parameters.
        """
        B, T, _ = x.shape  # B = batch size, T = sequence length (150)
        # 1. Project inputs to the model dimension
        x_emb = self.input_proj(x)  # shape: (B, T, d_model)
        # 2. Add positional encoding to embed sequence order
        x_emb = x_emb + self.pos_embedding  # add position information
        x_emb = self.dropout(x_emb)
        # 3. Pass through Transformer encoder layers (bidirectional self-attention over time)
        enc_out = self.transformer_encoder(x_emb) 
        # 4. Project each time step embedding to output pose dimensions
        output = self.output_proj(enc_out)  # shape: (B, T, output_dim)
        return output