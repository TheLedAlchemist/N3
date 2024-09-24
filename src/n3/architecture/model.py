import jax
import jax.numpy as jnp
import equinox as eqx

from jax.typing import ArrayLike
from jaxtyping import PRNGKeyArray, Float

from n3.architecture.controller import ControllerLike
from n3.utils.utils import control_to_mask


class N3(eqx.Module):
    in_size: int
    out_size: int
    layer_sizes: list[int]
    true_layer_sizes: list[int]
    layers: list[eqx.nn.Linear]

    def __init__(
        self,
        in_size,
        out_size,
        hidden_layer_sizes: list[int],
        key: PRNGKeyArray,
    ):
        self.in_size = in_size + 1  # additional input for size controller
        self.out_size = out_size
        self.layer_sizes = [self.in_size] + hidden_layer_sizes + [self.out_size]
        self.true_layer_sizes = (
            [self.in_size] + [1] * len(hidden_layer_sizes) + [self.out_size]
        )
        keys = jax.random.split(key, len(self.layer_sizes))
        self.layers = [
            eqx.nn.Linear(in_features, out_features, key=keys[key_idx])
            for key_idx, (in_features, out_features) in enumerate(
                zip(self.layer_sizes[:-1], self.layer_sizes[1:])
            )
        ]

    def __call__(
        self, x: ArrayLike, control: ControllerLike
    ) -> Float[jnp.ndarray, "out_size"]:
        control_value = control(jnp.ones((1,)))
        x_new = jnp.concatenate([x, control_value], axis=-1)
        for idx, layer in enumerate(self.layers[:-1]):
            mask = control_to_mask(control_value, self.true_layer_sizes[idx + 1])
            x_new = mask * jax.nn.tanh(layer(x_new))

        return self.layers[-1](x_new)
