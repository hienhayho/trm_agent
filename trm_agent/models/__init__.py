"""TRM model components."""

from .config import TRMConfig
from .embeddings import InputEmbedding, LatentEmbedding, TokenEmbedding, RoleEmbedding
from .ema import EMA, ModelEMA
from .heads import DecisionHead, ToolHead, UnifiedParamHead, QHead, ContentHead, OutputHead
from .layers import RMSNorm, SwiGLU, TransformerLayer, TRMBlock
from .trm import TRMForToolCalling, TRMOutput

__all__ = [
    # Config
    "TRMConfig",
    # Model
    "TRMForToolCalling",
    "TRMOutput",
    # Embeddings
    "InputEmbedding",
    "LatentEmbedding",
    "TokenEmbedding",
    "RoleEmbedding",
    # Layers
    "RMSNorm",
    "SwiGLU",
    "TransformerLayer",
    "TRMBlock",
    # Heads
    "DecisionHead",
    "ToolHead",
    "UnifiedParamHead",
    "QHead",
    "ContentHead",
    "OutputHead",
    # EMA
    "EMA",
    "ModelEMA",
]
