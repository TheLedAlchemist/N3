import numpy as np
import jax

from jaxtyping import Float, Array, PRNGKeyArray


class DataLoader:
    def __init__(
        self,
        x: Float[Array, "n_samples ..."],
        y: Float[Array, "n_samples ..."],
        batch_size: int,
        key: PRNGKeyArray,
        shuffle: bool = True,
    ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = x.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.key = key

    def __iter__(self):
        if self.shuffle:
            self.key, subkey = jax.random.split(self.key)
            indices = jax.random.permutation(subkey, self.num_samples)
        else:
            indices = np.arange(self.num_samples)

        for i in range(self.num_batches):
            batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.x[batch_indices], self.y[batch_indices]

    def __len__(self):
        return self.num_batches
