# Libraries
import jax
import optax
from flax.training import train_state
from matplotlib import pyplot as plt
import numpy as np

# Internal Imports
from src.misc.visualize_reconstructions import visualize_reconstructions
from src.data_loader.load_dataset import load_cifar10
from src.model.maeencoder import MAEEncoder
from src.model.maedecoder import MAEDecoder
from src.model.maeautoencoder import MaskedAutoencoder
from src.training.train_step import train_step
from src.testing.evaluate_rec import evaluate_reconstruction 
import jax.numpy as jnp

x_train, x_test = load_cifar10()

encoder = MAEEncoder()
decoder = MAEDecoder()
mae = MaskedAutoencoder(encoder=encoder, decoder=decoder)

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
params = mae.init({'params': init_rng, 'mask': init_rng}, jnp.ones((1, 32, 32, 3)))['params']

tx = optax.adam(learning_rate=1e-3)
state = train_state.TrainState.create(apply_fn=mae.apply, params=params, tx=tx)

batch_size = 128
num_epochs = 100
steps_per_epoch = len(x_train) // batch_size

losses = []  # List to store loss values

# Select a fixed set of test images for visualization
test_sample_idx = np.random.choice(len(x_test), 4, replace=False)
test_sample = x_test[test_sample_idx]

for epoch in range(num_epochs):
    total_loss = 0
    for step in range(steps_per_epoch):
        batch_idx = np.random.choice(len(x_train), batch_size)
        batch = x_train[batch_idx]
        rng, step_rng = jax.random.split(rng)
        state, loss, rng = train_step(state, batch, step_rng)
        total_loss += loss

    avg_loss = total_loss / steps_per_epoch
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    # Log to wandb
    

    # Visualize reconstructions every 10 epochs
    if (epoch + 1) % 10 == 0:
        rng, eval_rng = jax.random.split(rng)
        reconstructed, mask = evaluate_reconstruction(state, test_sample, eval_rng)
        visualize_reconstructions(test_sample, reconstructed, mask, epoch + 1)

print("Pretraining complete!")

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.savefig("loss_curve.png")
plt.close()