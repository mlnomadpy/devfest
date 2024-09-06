
import matplotlib.pyplot as plt

def visualize_reconstructions(original, reconstructed, mask, epoch):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    for i in range(4):
        # Original image
        axs[0, i].imshow(original[i])
        axs[0, i].set_title(f"Original {i+1}")
        axs[0, i].axis('off')

        # Reconstructed image
        axs[1, i].imshow(reconstructed[i])
        axs[1, i].set_title(f"Reconstructed {i+1}")
        axs[1, i].axis('off')

    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"reconstruction_epoch_{epoch}.png")
    plt.close()