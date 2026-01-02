"""TRM model components.

Note: Span extraction (slots/params) is handled by GLiNER2, not TRM.
TRM only handles decision classification and tool selection.
"""

from .config import TRMConfig
from .embeddings import InputEmbedding, LatentEmbedding, TokenEmbedding, RoleEmbedding
from .ema import EMA, ModelEMA
from .heads import DecisionHead, ToolHead, QHead, ContentHead, OutputHead
from .layers import RMSNorm, SwiGLU, ConvSwiGLU, TransformerLayer, TRMBlock
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
    "ConvSwiGLU",
    "TransformerLayer",
    "TRMBlock",
    # Heads
    "DecisionHead",
    "ToolHead",
    "QHead",
    "ContentHead",
    "OutputHead",
    # EMA
    "EMA",
    "ModelEMA",
]
