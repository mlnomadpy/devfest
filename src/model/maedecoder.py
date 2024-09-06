from flax import linen as nn
from src.model.vit import ViT

class MAEDecoder(nn.Module):
    patch_size: int = 4
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 256

    @nn.compact
    def __call__(self, x, mask):
        b, n = mask.shape

        # Add position embedding
        pos_embedding = self.param('pos_embedding', nn.initializers.normal(stddev=0.02), (1, n, self.hidden_dim))
        x = nn.Dense(features=self.hidden_dim)(x)

        # Add mask tokens
        mask_token = self.param('mask_token', nn.initializers.normal(stddev=0.02), (1, 1, self.hidden_dim))
        mask_tokens = jnp.broadcast_to(mask_token, (b, n, self.hidden_dim))
        x = x * mask[:, :, None] + mask_tokens * (1 - mask[:, :, None])

        x = x + pos_embedding

        # Transformer blocks
        vit = ViT(patch_size=self.patch_size, hidden_dim=self.hidden_dim,
                  num_heads=self.num_heads, num_layers=self.num_layers, mlp_dim=self.mlp_dim)
        x = vit(x)

        # Project to patch dimension
        x = nn.Dense(features=self.patch_size**2 * 3)(x)
        x = nn.sigmoid(x)

        return x