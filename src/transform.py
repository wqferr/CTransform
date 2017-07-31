from PIL import Image
from functools import partial, reduce
import numpy as np


def _apply_func_to_img(img, dim_in, dim_out, xlim, ylim, pixels, cur, nxt):
    pos, z = cur
    r0 = pos % dim_in[0]
    c0 = (pos - r0) / dim_in[0]
    r1 = round(dim_out[1] * (np.imag(z) - ylim[0]) / (ylim[1] - ylim[0]))
    c1 = round(dim_out[0] * (np.real(z) - xlim[0]) / (xlim[1] - xlim[0]))
    if 0 <= c1 < dim_out[0] and 0 <= r1 < dim_out[1]:
        p = img.getpixel((r0, c0))
        pixels[int(c1)][int(r1)].append(p)
    return nxt


def apply_to(func, img, *,
             xlim_in=(-1, 1), xlim_out=(-1, 1),
             ylim_in=(-1, 1), ylim_out=(-1, 1),
             lim_in=None, lim_out=None,
             dim_out=None):
    if lim_in is not None:
        xlim_in = lim_in
        ylim_in = lim_in

    if lim_out is not None:
        xlim_out = lim_out
        ylim_out = lim_out

    dim_in = img.size
    if dim_out is None:
        dim_out = dim_in

    newImg = Image.new(img.mode, dim_out)
    pixels = newImg.load()
    pixList = [[[] for j in range(dim_out[1])] for i in range(dim_out[0])]

    x = np.linspace(xlim_in[0], xlim_in[1], dim_in[0])
    y = np.linspace(ylim_in[0], ylim_in[1], dim_in[1]) * 1j
    z0 = np.sum(
        np.transpose([np.tile(x, dim_in[1]), np.repeat(y, dim_in[0])]),
        axis=1
    )
    z1 = func(z0)
    reduce(
        partial(_apply_func_to_img, img, dim_in,
                dim_out, xlim_out, ylim_out, pixList),
        enumerate(z1)
    )

    for i, r in enumerate(pixList):
        for j, p in enumerate(r):
            if p:
                a = np.array(p)
                r, g, b = a[:, 0], a[:, 1], a[:, 2]
                pixels[i, j] = (
                    min(int(round(np.sum(r))), 255),
                    min(int(round(np.sum(g))), 255),
                    min(int(round(np.sum(b))), 255)
                )

    return newImg
