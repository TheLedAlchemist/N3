import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.special import jv


def bessel(x):
    """
    J_0 + J_1 + J_2
    """
    return jv(0, x) + jv(1, x) + jv(2, x)


def generate_data(
    n_samples: int = 1000,
    test_size: float = 0.2,
    scaler: MinMaxScaler = MinMaxScaler(feature_range=(-1, 1)),
    seed: int = 0,
):
    x = np.linspace(-2 * np.pi, 2 * np.pi, n_samples).reshape(-1, 1)
    y = bessel(x).reshape(-1, 1)

    x_scaled, y_scaled = scaler.fit_transform(x), scaler.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_scaled, test_size=test_size, random_state=seed
    )

    return jnp.array(x_train), jnp.array(x_test), jnp.array(y_train), jnp.array(y_test)
