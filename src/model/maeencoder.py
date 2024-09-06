from flax import linen as nn
from src.model.vit import ViT

class MAEEncoder(nn.Module):
    patch_size: int = 4
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    mlp_dim: int = 512

    @nn.compact
    def __call__(self, x, mask):
        # x is already patchified
        b, n, c = x.shape
        pos_embedding = self.param('pos_embedding', nn.initializers.normal(stddev=0.02), (1, n, self.hidden_dim))

        # Add position embedding to unmasked tokens
        x = nn.Dense(features=self.hidden_dim)(x)
        x = x + pos_embedding

        # Apply mask
        x = x * mask[:, :, None]

        # Transformer blocks
        vit = ViT(patch_size=self.patch_size, hidden_dim=self.hidden_dim,
                  num_heads=self.num_heads, num_layers=self.num_layers, mlp_dim=self.mlp_dim)
        x = vit(x)

        return x