import numpy as np


# in this file, we modify s to be the inverse of s
def sigmoid(x, s=1):
    return 1 / (1 + np.exp(-x / s))


def s_density(x, sdf, s=1, dx_dt=1):
    e_sx = np.exp(-sdf / s)
    return (e_sx / s) / (1 + e_sx) ** 2


def opaque_density(x, sdf, s=1, dx_dt=1):
    sigmoid_sdf = sigmoid(sdf, s)
    derivative = np.gradient(sigmoid_sdf, x)
    return np.maximum((-derivative * dx_dt) / sigmoid_sdf, 0)


def beta_density(x, sdf, beta=0.1, dx_dt=1):
    beta += 1e-8
    return (0.5 + 0.5 * np.sign(sdf) * np.expm1(-np.abs(sdf) / beta)) / beta


def transmittance(x, density, dx=None):
    if dx is None:
        dx = np.gradient(x)
    integral = np.cumsum(dx * density)
    integral = np.concatenate(([0], integral[:-1]))
    return np.exp(-integral)


def weight_density(T, density, alpha):
    return T * density


def weight_alpha(T, density, alpha):
    return T * alpha


def normalized_density(x, density, dx=None):
    if dx is None:
        dx = np.gradient(x)
    integral = np.sum(density * dx)
    return density / (integral + 1e-8)


def alpha(x, density, dx=None):
    if dx is None:
        dx = np.gradient(x)
    return 1 - np.exp(-density * dx)


def transmittance_prod(alpha):
    return np.cumprod(1 - alpha)


densities = {
    "s_density": s_density,
    "beta_density": beta_density,
    "opaque_density": opaque_density,
}
weights = {"weight_density": weight_density, "weight_alpha": weight_alpha}
