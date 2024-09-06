import jax
import optax

@jax.jit
def train_step(state, batch, rng):
    rng, new_rng = jax.random.split(rng)
    def loss_fn(params):
        reconstructed, mask = state.apply_fn({'params': params}, batch, train=True, rngs={'mask': rng})
        loss = optax.l2_loss(reconstructed, batch).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, new_rng

