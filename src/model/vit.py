from flax import linen as nn
from src.model.transformer_block import TransformerBlock

class ViT(nn.Module):
    patch_size: int = 4
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    mlp_dim: int = 512

    @nn.compact
    def __call__(self, x, train=True):
        b, n, c = x.shape

        # Add position embedding
        pos_embedding = self.param('pos_embedding', nn.initializers.normal(stddev=0.02), (1, n, self.hidden_dim))
        x = nn.Dense(features=self.hidden_dim)(x)
        x = x + pos_embedding

        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(num_heads=self.num_heads, mlp_dim=self.mlp_dim)(x)

        return x