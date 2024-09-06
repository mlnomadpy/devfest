def evaluate_reconstruction(state, test_images, rng):
    reconstructed, mask = state.apply_fn({'params': state.params}, test_images, train=False, rngs={'mask': rng})
    return reconstructed, mask
