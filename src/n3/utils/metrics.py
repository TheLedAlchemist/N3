import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Float, Array, Int

from n3.architecture.model import ModelLike
from n3.architecture.controller import ControllerLike


@eqx.filter_jit()
def accuracy(
    model: ModelLike,
    control: ControllerLike,
    x: Float[Array, "batch 2"],
    y: Int[Array, "batch"],
) -> Float[Array, ""]:
    logits = jax.vmap(model, in_axes=(0, None))(x, control)
    return jnp.mean(jnp.argmax(logits, axis=-1) == y)


@eqx.filter_jit
def cross_entropy(
    y: Int[Array, "batch"], pred_y: Float[Array, "batch num_classes"]
) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-9.
    # pred_y are the log-softmax'd predictions.
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)
