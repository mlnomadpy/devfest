from flax import linen as nn
import jax
import jax.numpy as jnp


class MaskedAutoencoder(nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    mask_ratio: float = 0.75
    patch_size: int = 4

    @nn.compact
    def __call__(self, img, train=True, rngs=None):
        # Patchify the image (for CIFAR-10, we'll use 4x4 patches)
        patches = self.patchify(img)
        batch, num_patches, _ = patches.shape

        # Create mask
        mask = self.create_mask(rngs['mask'] if rngs is not None else None, batch, num_patches)

        # Encode
        encoded = self.encoder(patches, mask)

        # Decode
        decoded = self.decoder(encoded, mask)

        # Unpatchify the output
        reconstructed = self.unpatchify(decoded, img.shape)

        return reconstructed, mask

    def patchify(self, imgs):
        batch_size, height, width, channels = imgs.shape
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        patches = imgs.reshape(batch_size, height // self.patch_size, self.patch_size, width // self.patch_size, self.patch_size, channels)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(batch_size, num_patches, -1)
        return patches

    def unpatchify(self, patches, original_shape):
        batch_size, height, width, channels = original_shape
        patches = patches.reshape(batch_size, height // self.patch_size, width // self.patch_size, self.patch_size, self.patch_size, channels)
        imgs = patches.transpose(0, 1, 3, 2, 4, 5)
        imgs = imgs.reshape(batch_size, height, width, channels)
        return imgs

    def create_mask(self, rng, batch, num_patches):
        if rng is None:
            rng = self.make_rng('mask')
        noise = jax.random.uniform(rng, (batch, num_patches))
        mask = jnp.where(noise > self.mask_ratio, 1., 0.)
        return mask