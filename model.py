"""Vision Transformer (ViT) implementation for CIFAR-10"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Project to patch embeddings: (B, embed_dim, num_patches_h, num_patches_w)
        x = self.projection(x)
        
        # Flatten: (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose: (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""
    
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm."""
    
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)
        
    def forward(self, x):
        # Pre-norm attention with residual
        x = x + self.attn(self.norm1(x))
        # Pre-norm MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer model for image classification."""
    
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, dim
        )
        
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer encoder
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        # Truncated normal for positional embeddings (ViT style)
        nn.init.trunc_normal_(self.patch_embed.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.cls_token, std=0.02)
                
    def forward(self, x):
        # Get patch embeddings
        x = self.patch_embed(x)
        x = self.dropout(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Use CLS token for classification
        x = self.norm(x)
        cls_token = x[:, 0]
        
        return self.classifier(cls_token)


if __name__ == "__main__":
    # Test the model
    from config import Config
    cfg = Config()
    
    model = VisionTransformer(
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_channels=3,
        num_classes=cfg.num_classes,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout,
        emb_dropout=cfg.emb_dropout
    )
    
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
