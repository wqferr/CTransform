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
        p = img.getpixel((c0, r0))
        if np.sum(p) > 0:
            pixels[c1, r1] = p
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

    if dim_out is None:
        dim_out = img.size

    dim_in = img.size
    newImg = Image.new(img.mode, dim_out)
    pixels = newImg.load()

    x = np.linspace(xlim_in[0], xlim_in[1], dim_in[0])
    y = np.linspace(ylim_in[0], ylim_in[1], dim_in[1]) * 1j
    z0 = np.sum(
        np.transpose([np.tile(x, dim_in[1]), np.repeat(y, dim_in[0])]),
        axis=1
    )
    z1 = func(z0)
    reduce(
        partial(_apply_func_to_img, img, dim_in,
                dim_out, xlim_out, ylim_out, pixels),
        enumerate(z1)
    )

    return newImg
