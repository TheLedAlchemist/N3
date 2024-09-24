import jax
import jax.numpy as jnp
import equinox as eqx

from jax.typing import ArrayLike
from jaxtyping import PRNGKeyArray
from typing import Protocol, runtime_checkable, Any
from beartype import beartype


@runtime_checkable
class ControllerLike(Protocol):
    params: Any

    def __call__(self, x: ArrayLike) -> ArrayLike: ...


class controller(eqx.Module):
    dim: int
    params: jax.Array

    def __init__(self, dim: int, key: PRNGKeyArray):
        self.dim = dim
        self.params = jax.random.normal(key, (dim, dim)) * 1e-5

    @beartype
    def __call__(self, x: ArrayLike) -> jnp.ndarray:
        return jnp.dot(x, self.params)
