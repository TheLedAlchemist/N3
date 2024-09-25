import jax
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from jaxtyping import Array, PRNGKeyArray, Float


def create_spiral_dataset(
    key: PRNGKeyArray, n_samples: int, num_classes: int
) -> tuple[Float[Array, "n_samples 2"], Float[Array, "n_samples"]]:
    """
    Create a multi-class 2D spiral dataset.

    Parameters:
    - key: JAX random key.
    - n_samples: Dataset size
    - num_classes: Number of spiral arms/classes.

    Returns:
    - points: The dataset points.
    - labels: The labels for each point.
    """
    points = []
    labels = []
    points_per_class = n_samples // num_classes

    for class_idx in range(num_classes):
        # Adjust the random key for each class
        key, subkey = jax.random.split(key)
        # Generate radii and angles for the current class
        radii = jnp.linspace(0.0, 1, points_per_class) * 5
        angles = (
            jnp.linspace(class_idx * 4, (class_idx + 1) * 4, points_per_class)
            + jax.random.normal(subkey, (points_per_class,)) * 0.2
        )
        # Convert polar to Cartesian
        x = radii * jnp.sin(angles)
        y = radii * jnp.cos(angles)
        class_points = jnp.column_stack((x, y))
        points.append(class_points)
        labels.append(jnp.full(points_per_class, class_idx))
    # Concatenate all class points and labels
    points = jnp.vstack(points)
    labels = jnp.hstack(labels)
    return points, labels


def generate_data(
    n_samples: int = 1000,
    num_classes: int = 5,
    test_size: float = 0.2,
    scaler: MinMaxScaler = MinMaxScaler(feature_range=(-1, 1)),
    seed: int = 0,
):
    key = jax.random.PRNGKey(seed)
    x, labels = create_spiral_dataset(key, n_samples, num_classes)

    x_scaled = scaler.fit_transform(np.array(x))

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, labels, test_size=test_size, random_state=seed
    )

    return jnp.array(x_train), jnp.array(x_test), jnp.array(y_train), jnp.array(y_test)
