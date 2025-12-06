# %%
{
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}



# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def perlin(width, height, scale=8, seed=None):
    rng = np.random.default_rng(seed)

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(a, b, t):
        return a + t * (b - a)

    gx = width // scale + 2
    gy = height // scale + 2

    angles = rng.random((gy, gx)) * 2 * np.pi
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    y, x = np.mgrid[0:height, 0:width]
    xf = x / scale
    yf = y / scale

    x0 = np.floor(xf).astype(int)
    y0 = np.floor(yf).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    sx = xf - x0
    sy = yf - y0

    g00 = gradients[y0, x0]
    g10 = gradients[y0, x1]
    g01 = gradients[y1, x0]
    g11 = gradients[y1, x1]

    d00 = np.dstack((sx,     sy    ))
    d10 = np.dstack((sx - 1, sy    ))
    d01 = np.dstack((sx,     sy - 1))
    d11 = np.dstack((sx - 1, sy - 1))

    n00 = np.sum(g00 * d00, axis=2)
    n10 = np.sum(g10 * d10, axis=2)
    n01 = np.sum(g01 * d01, axis=2)
    n11 = np.sum(g11 * d11, axis=2)

    u = fade(sx)
    v = fade(sy)

    nx0 = lerp(n00, n10, u)
    nx1 = lerp(n01, n11, u)
    nxy = lerp(nx0, nx1, v)

    nmin, nmax = nxy.min(), nxy.max()
    return (nxy - nmin) / (nmax - nmin + 1e-12)

# %%

width, height = 512, 512
img = perlin(width, height, scale=16, seed=42)

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title('Perlin Noise')
plt.show()

# %%

# "jupyter": "This is a Jupyter notebook.  Each cell is delimited by # %%.  Please provide some debugging output as each cell starts to run and complete. Please ensure that any code you provide is valid Python code that can run in a Jupyter notebook environment or as a single python script"   
