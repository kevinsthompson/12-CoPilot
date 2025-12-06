# %%
{
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}


# %% [markdown]
# # Perlin Noise Demo
# This notebook generates and displays a 2D Perlin noise image using NumPy and Matplotlib.


# %%
print("[Cell] Importing libraries...")
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
print("[Cell] Imports complete.")


# %%
def perlin(width, height, scale=8, seed=None):
    print(f"[Cell] perlin() start: width={width}, height={height}, scale={scale}, seed={seed}")

    # Seeded RNG for reproducibility
    rng = np.random.default_rng(seed)

    # Smoothstep-like fade function: improves continuity of derivatives
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    # Linear interpolation between a and b with weight t
    def lerp(a, b, t):
        return a + t * (b - a)

    # Gradient grid dimensions (extra border to avoid bounds checks)
    gx = width // scale + 2
    gy = height // scale + 2

    # Random unit gradient vectors per grid point using random angles
    angles = rng.random((gy, gx)) * 2 * np.pi
    gradients = np.dstack((np.cos(angles), np.sin(angles)))  # shape: (gy, gx, 2)

    # Pixel coordinates; map to grid space
    y, x = np.mgrid[0:height, 0:width]
    xf = x / scale
    yf = y / scale

    # Indices of the four corners of the cell containing each pixel
    x0 = np.floor(xf).astype(int)
    y0 = np.floor(yf).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Relative position of pixel within its cell [0,1)
    sx = xf - x0
    sy = yf - y0

    # Gradient vectors at the four corners
    g00 = gradients[y0, x0]
    g10 = gradients[y0, x1]
    g01 = gradients[y1, x0]
    g11 = gradients[y1, x1]

    # Displacement vectors from each corner to the pixel
    d00 = np.dstack((sx,     sy    ))
    d10 = np.dstack((sx - 1, sy    ))
    d01 = np.dstack((sx,     sy - 1))
    d11 = np.dstack((sx - 1, sy - 1))

    # Corner contributions via dot products (gradient • displacement)
    n00 = np.sum(g00 * d00, axis=2)
    n10 = np.sum(g10 * d10, axis=2)
    n01 = np.sum(g01 * d01, axis=2)
    n11 = np.sum(g11 * d11, axis=2)

    # Smooth interpolation weights
    u = fade(sx)
    v = fade(sy)

    # Interpolate along x for top and bottom, then along y
    nx0 = lerp(n00, n10, u)
    nx1 = lerp(n01, n11, u)
    nxy = lerp(nx0, nx1, v)

    # Normalize to [0,1] for display
    nmin, nmax = nxy.min(), nxy.max()
    result = (nxy - nmin) / (nmax - nmin + 1e-12)

    print("[Cell] perlin() complete.")
    return result


# %%
print("[Cell] Building interactive widgets...")

width_w = widgets.IntSlider(value=512, min=64, max=1024, step=32, description='Width')
height_w = widgets.IntSlider(value=512, min=64, max=1024, step=32, description='Height')
scale_w = widgets.IntSlider(value=16, min=2, max=128, step=2, description='Scale')
seed_w = widgets.IntText(value=42, description='Seed')
cmap_w = widgets.Dropdown(options=['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'], value='gray', description='Colormap')
interp_w = widgets.Dropdown(options=['nearest', 'bilinear', 'bicubic'], value='nearest', description='Interpolation')

out = widgets.Output()

def render(_=None):
    print("[Cell] Render start.")
    with out:
        clear_output(wait=True)
        img = perlin(width_w.value, height_w.value, scale=scale_w.value, seed=seed_w.value)
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap=cmap_w.value, interpolation=interp_w.value)
        plt.axis('off')
        plt.title(f'Perlin Noise (scale={scale_w.value}, seed={seed_w.value})')
        plt.show()
    print("[Cell] Render complete.")

for w in (width_w, height_w, scale_w, seed_w, cmap_w, interp_w):
    w.observe(render, names='value')

controls = widgets.VBox([width_w, height_w, scale_w, seed_w, cmap_w, interp_w])
ui = widgets.HBox([controls, out])

display(ui)
render()

print("[Cell] Interactive setup complete.")

# %%
# "jupyter": "This is a Jupyter notebook.  Each cell is delimited by # %%.  Please provide some debugging output as each cell starts to run and complete. Please ensure that any code you provide is valid Python code that can run in a Jupyter notebook environment or as a single python script"
