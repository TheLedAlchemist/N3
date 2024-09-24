from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Float, Array, PyTree


@partial(jax.jit, static_argnames="N")
def control_to_mask(control_value: Float[Array, ""], N: int) -> Float[Array, "N"]:
    """
    Maps the control value to a vector mask of size N.
    The sigmoid of control specifies the number of elements in the vector to set to 1 and 0.
    The intermediate element takes a fractional value.
    """
    # Compute mask_param using vectorized operations
    mask_param = jnp.sin((jnp.pi / 2) * control_value**2) ** 2
    total_full_ones = jnp.floor(mask_param * N).astype(int)
    fractional_part = mask_param * N - total_full_ones

    # Ensure num_ones is within range [0, N]
    # num_ones = jax.lax.clamp(0, total_full_ones, N).astype(int)
    num_ones = jnp.clip(total_full_ones, 0, N)

    mask = jnp.zeros(N)

    # Use lax.select to create a 1 array of size N and set the first `num_ones` elements to 1
    ones_mask = jnp.arange(N) < num_ones

    # Apply ones_mask to the mask
    mask = jax.lax.select(ones_mask, jnp.ones(N), mask)

    # Set the fractional part at the boundary element if `num_ones < N`
    mask = jax.lax.cond(
        jnp.squeeze(num_ones < N),
        lambda m: m.at[num_ones].set(fractional_part),
        lambda m: m,
        mask,
    )

    return mask


@eqx.filter_jit()
def grad_norm(grads: PyTree) -> Float[Array, ""]:
    return jnp.sqrt(
        sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(grads))
    )
