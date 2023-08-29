import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt

from sdfs import sdfs
from weights import (
    densities,
    weights,
    transmittance,
    transmittance_prod,
    alpha,
    normalized_density,
)

custom_sdf = None


def visualize_gt(
    x,
    sdf=None,
    rgb=None,
    density=None,
    alpha=None,
    transmittance=None,
    weight=None,
    fig=None,
    ax=None,
    line_type="-",
):
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    if sdf is not None:
        # ax.plot(x, sdf, "k" + line_type, label="sdf", c=rgb)
        if rgb is None:
            ax.plot(x, sdf, "k" + line_type, label="sdf")
        else:
            ax.scatter(x, sdf, c=rgb, label="sdf", s=20, edgecolors="k")
    if density is not None:
        ax.plot(x, density, "r" + line_type, label="density")
    if alpha is not None:
        ax.plot(x, alpha, "y" + line_type, label="alpha")
    if transmittance is not None:
        ax.plot(x, transmittance, "b" + line_type, label="transmittance")
    if weight is not None:
        ax.plot(x, weight, "g" + line_type, label="weight")

    ax.legend()
    ax.grid(True)

    return fig, ax


def visualize(
    x=np.linspace(-1, 1, 100),
    dx=None,
    sdf=sdfs["one_point"],
    density=densities["beta_density"],
    density_param=1.0,
    normalize_density=False,
    weight=weights["weight_alpha"],
    fig=None,
    ax=None,
    line_type="-",
):
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    if callable(sdf):
        sdf_val = sdf(x)
    else:
        sdf_val = sdf
    sdfs["user_input"] = sdf

    if dx is None:
        dx = np.gradient(x)

    density_val = density(x, sdf_val, density_param)
    alpha_val = alpha(x, density_val, dx)
    transmittance_val = transmittance(x, density_val, dx)
    weight_val = weight(transmittance_val, density_val, alpha_val)

    (sdf_line,) = ax.plot(x, sdf_val, "k" + line_type, label="sdf")
    (density_line,) = ax.plot(x, density_val, "r" + line_type, label="density")
    (alpha_line,) = ax.plot(x, alpha_val, "y" + line_type, label="alpha")
    (transmittance_line,) = ax.plot(
        x, transmittance_val, "b" + line_type, label="transmittance"
    )
    (weight_line,) = ax.plot(x, weight_val, "g" + line_type, label="weight")

    ax.legend()
    ax.grid(True)

    def update(
        density_param=widgets.FloatLogSlider(
            base=10, min=-6, max=2, step=0.01, value=density_param
        ),
        sdf_key=widgets.Dropdown(options=sdfs.keys(), value="user_input"),
        density=widgets.Dropdown(options=densities, value=density),
        normalize_density=widgets.Checkbox(value=normalize_density),
        weight=widgets.Dropdown(options=weights, value=weight),
    ):
        sdf = sdfs[sdf_key]
        if callable(sdf):
            sdf_val = sdf(x)
        else:
            sdf_val = sdf

        density_val = density(x, sdf_val, density_param)
        if normalize_density:
            density_val = normalized_density(x, density_val, dx)
        alpha_val = alpha(x, density_val, dx)
        transmittance_val = transmittance(x, density_val, dx)
        weight_val = weight(transmittance_val, density_val, alpha_val)

        sdf_line.set_ydata(sdf_val)
        density_line.set_ydata(density_val)
        alpha_line.set_ydata(alpha_val)
        transmittance_line.set_ydata(transmittance_val)
        weight_line.set_ydata(weight_val)

        fig.canvas.draw_idle()

    _ = widgets.interact(update)
    return fig, ax
