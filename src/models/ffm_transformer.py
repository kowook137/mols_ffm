#!/usr/bin/env python3
"""
Fisher-Flow Grid-Transformer for MOLS Generation
RTX 3080 듀얼 GPU 최적화된 2D Grid Attention 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Path 설정 (상대 import 문제 해결)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.datasets.latin_dataset import FisherGeometry


@dataclass
class ModelConfig:
    """Fisher-Flow 모델 설정"""
    # 모델 구조
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    
    # Grid-Transformer 특화
    max_order: int = 9
    use_row_col_attention: bool = True
    use_positional_encoding: bool = True
    
    # Fisher-Flow 특화
    sphere_embedding_dim: int = 512
    tangent_projection: bool = True
    fisher_distance_weight: float = 1.0
    
    # GPU 최적화
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    fused_adam: bool = True


class SinusoidalPositionalEncoding(nn.Module):
    """2D Sinusoidal positional encoding for grids"""
    
    def __init__(self, d_model: int, max_order: int = 9):
        super().__init__()
        self.d_model = d_model
        self.max_order = max_order
        
        # Create 2D positional encoding
        pe = torch.zeros(max_order, max_order, d_model)
        
        for i in range(max_order):
            for j in range(max_order):
                for k in range(0, d_model, 4):
                    # Row position encoding
                    pe[i, j, k] = math.sin(i / (10000 ** (k / d_model)))
                    pe[i, j, k + 1] = math.cos(i / (10000 ** (k / d_model)))
                    
                    # Column position encoding  
                    if k + 2 < d_model:
                        pe[i, j, k + 2] = math.sin(j / (10000 ** ((k + 2) / d_model)))
                    if k + 3 < d_model:
                        pe[i, j, k + 3] = math.cos(j / (10000 ** ((k + 2) / d_model)))
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, order, order, d_model]
        Returns:
            x + positional encoding
        """
        order = x.size(1)
        return x + self.pe[:order, :order, :].unsqueeze(0)


class GridMultiHeadAttention(nn.Module):
    """2D Grid Multi-Head Attention with row/column factorization"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Attention projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Separate row and column attention
        self.row_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.col_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, order, order, d_model]
            mask: Optional attention mask
        Returns:
            Attended features [batch, order, order, d_model]
        """
        batch_size, order, _, d_model = x.shape
        
        # Row-wise attention
        x_rows = x.view(batch_size * order, order, d_model)  # [batch*order, order, d_model]
        row_out, _ = self.row_attention(x_rows, x_rows, x_rows)
        row_out = row_out.view(batch_size, order, order, d_model)
        
        # Column-wise attention  
        x_cols = x.transpose(1, 2).contiguous().view(batch_size * order, order, d_model)
        col_out, _ = self.col_attention(x_cols, x_cols, x_cols)
        col_out = col_out.view(batch_size, order, order, d_model).transpose(1, 2)
        
        # Combine row and column attention
        output = self.output_proj(row_out + col_out)
        return self.dropout(output)


class FisherFlowBlock(nn.Module):
    """Fisher-Flow Transformer Block with Grid Attention"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Grid attention
        self.grid_attention = GridMultiHeadAttention(
            config.d_model, config.n_heads, config.dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, order, order, d_model]
        Returns:
            Transformed features [batch, order, order, d_model]
        """
        # Grid attention with residual
        attn_out = self.grid_attention(self.norm1(x), mask)
        x = x + attn_out
        
        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x


class SphereEmbedding(nn.Module):
    """Embedding layer for Fisher-Rao sphere points"""
    
    def __init__(self, max_order: int, d_model: int):
        super().__init__()
        self.max_order = max_order
        self.d_model = d_model
        
        # Sphere point → embedding projection
        self.sphere_proj = nn.Linear(max_order, d_model)
        
        # Learnable order embedding
        self.order_embedding = nn.Embedding(max_order + 1, d_model)
        
    def forward(self, sphere_points: torch.Tensor, order: int) -> torch.Tensor:
        """
        Args:
            sphere_points: [batch, order, order, order] (Fisher-Rao points)
            order: Size of Latin Square
        Returns:
            Embedded features [batch, order, order, d_model]
        """
        batch_size, H, W, D = sphere_points.shape
        
        # Project sphere points to embedding space
        embeddings = self.sphere_proj(sphere_points)  # [batch, order, order, d_model]
        
        # Add order information
        order_emb = self.order_embedding(torch.tensor(order, device=sphere_points.device))
        embeddings = embeddings + order_emb.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        return embeddings


class TangentProjection(nn.Module):
    """Tangent space projection for Fisher-Flow"""
    
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        
        self.projection = nn.Linear(d_model, output_dim)
        self.geometry = FisherGeometry()
        
    def forward(self, features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Tangent space projection: ṽ - ⟨x, ṽ⟩x
        
        Args:
            features: [batch, order, order, d_model]
            x: Current sphere points [batch, order, order, order]
        Returns:
            Tangent vectors [batch, order, order, order]
        """
        # Project to output dimension
        v_tilde = self.projection(features)  # [batch, order, order, output_dim]
        
        # Flatten for tangent projection
        batch_shape = v_tilde.shape[:-1]
        v_flat = v_tilde.view(-1, v_tilde.shape[-1])
        x_flat = x.view(-1, x.shape[-1])
        
        # Tangent projection: ṽ - ⟨x, ṽ⟩x
        dot_product = torch.sum(x_flat * v_flat, dim=-1, keepdim=True)
        tangent_v = v_flat - dot_product * x_flat
        
        # Reshape back
        tangent_v = tangent_v.view(batch_shape + (v_tilde.shape[-1],))
        
        return tangent_v


class FisherFlowTransformer(nn.Module):
    """
    Fisher-Flow Grid-Transformer for MOLS Generation
    RTX 3080 듀얼 GPU 최적화
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.geometry = FisherGeometry()
        
        # Embedding layers
        self.sphere_embedding = SphereEmbedding(config.max_order, config.d_model)
        
        # Positional encoding
        if config.use_positional_encoding:
            self.pos_encoding = SinusoidalPositionalEncoding(config.d_model, config.max_order)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            FisherFlowBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output projection to tangent space
        self.tangent_projection = TangentProjection(config.d_model, config.max_order)
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(config.d_model)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        sphere_points: torch.Tensor,
        t: torch.Tensor,
        order: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for Fisher-Flow matching
        
        Args:
            sphere_points: [batch, order, order, order] Fisher-Rao points
            t: [batch] Time parameter for flow matching
            order: Size of Latin Square
            mask: Optional attention mask
            
        Returns:
            Velocity field on tangent space [batch, order, order, order]
        """
        batch_size = sphere_points.size(0)
        
        # Embedding
        x = self.sphere_embedding(sphere_points, order)  # [batch, order, order, d_model]
        
        # Add positional encoding
        if hasattr(self, 'pos_encoding'):
            x = self.pos_encoding(x)
        
        # Time embedding (broadcast)
        t_emb = torch.sin(t.unsqueeze(-1) * math.pi)  # [batch, 1]
        t_features = t_emb.unsqueeze(1).unsqueeze(1).expand(-1, order, order, -1)
        x = torch.cat([x, t_features], dim=-1)
        
        # Adjust dimension if needed
        if x.size(-1) != self.config.d_model:
            x = nn.Linear(x.size(-1), self.config.d_model, device=x.device)(x)
        
        # Transformer blocks
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, mask)
            else:
                x = block(x, mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to tangent space
        velocity = self.tangent_projection(x, sphere_points)
        
        return velocity
    
    def get_model_size(self) -> Dict[str, int]:
        """모델 크기 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # float32 기준
        }


def create_model(config: ModelConfig, device_ids: Optional[list] = None) -> nn.Module:
    """
    RTX 3080 듀얼 GPU를 위한 모델 생성
    
    Args:
        config: 모델 설정
        device_ids: GPU device IDs (e.g., [0, 1])
        
    Returns:
        DDP wrapped model or single GPU model
    """
    model = FisherFlowTransformer(config)
    
    if device_ids is not None and len(device_ids) > 1:
        # Multi-GPU setup
        if not dist.is_initialized():
            print("Warning: DDP not initialized, using DataParallel instead")
            model = nn.DataParallel(model, device_ids=device_ids)
        else:
            model = DDP(model, device_ids=device_ids, output_device=device_ids[0])
        
        print(f"Model distributed across GPUs: {device_ids}")
    else:
        # Single GPU
        device = torch.device(f'cuda:{device_ids[0]}' if device_ids else 'cuda')
        model = model.to(device)
        print(f"Model on single GPU: {device}")
    
    # Print model info
    model_info = model.module.get_model_size() if hasattr(model, 'module') else model.get_model_size()
    print(f"Model parameters: {model_info['total_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.1f} MB")
    
    return model


# Gradient clipping을 위한 유틸리티
def clip_grad_norm(model: nn.Module, max_norm: float = 1.0):
    """Gradient clipping with proper handling of DDP"""
    if hasattr(model, 'module'):
        return torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm)
    else:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# 예제 사용법
if __name__ == "__main__":
    # 설정
    config = ModelConfig(
        d_model=512,
        n_layers=6,
        n_heads=8,
        max_order=9,
        gradient_checkpointing=True
    )
    
    # 모델 생성 (RTX 3080 듀얼 GPU)
    model = create_model(config, device_ids=[0, 1])
    
    # 테스트 데이터
    batch_size = 4
    order = 8
    device = torch.device('cuda:0')
    
    sphere_points = torch.randn(batch_size, order, order, order, device=device)
    sphere_points = F.normalize(sphere_points, p=2, dim=-1)  # Unit sphere
    t = torch.rand(batch_size, device=device)
    
    # Forward pass
    with torch.cuda.amp.autocast(enabled=config.mixed_precision):
        velocity = model(sphere_points, t, order)
    
    print(f"Input shape: {sphere_points.shape}")
    print(f"Output shape: {velocity.shape}")
    print(f"GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB") 