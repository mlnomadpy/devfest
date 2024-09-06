from flax import linen as nn
from src.model.attention import Attention
from src.model.mlp import MLP

class TransformerBlock(nn.Module):
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = Attention(num_heads=self.num_heads)(y)
        x = x + y

        y = nn.LayerNorm()(x)
        y = MLP(hidden_dim=self.mlp_dim, out_dim=x.shape[-1])(y)
        return x + y