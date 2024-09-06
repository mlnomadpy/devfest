from flax import linen as nn
from jax import numpy as jnp

class Attention(nn.Module):
    num_heads: int

    @nn.compact
    def __call__(self, x):
        d_model = x.shape[-1]
        d_head = d_model // self.num_heads

        qkv = nn.Dense(features=d_model * 3, use_bias=False)(x)
        qkv = qkv.reshape(x.shape[0], -1, 3, self.num_heads, d_head)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(d_head)
        attention = nn.softmax(attention, axis=-1)

        y = jnp.matmul(attention, v)
        y = jnp.transpose(y, (0, 2, 1, 3))
        y = y.reshape(x.shape[0], -1, d_model)

        y = nn.Dense(features=d_model)(y)
        return y