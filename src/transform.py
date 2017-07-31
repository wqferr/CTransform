from PIL import Image
from functools import partial, reduce
import numpy as np


def _apply_func_to_img(img, w, h, xlim, ylim, pixels, cur, nxt):
    pos, z = cur
    r0 = pos % w
    c0 = (pos - r0) / w
    r1 = round(h * (np.imag(z) - ylim[0]) / (ylim[1] - ylim[0]))
    c1 = round(w * (np.real(z) - xlim[0]) / (xlim[1] - xlim[0]))
    if 0 <= c1 < w and 0 <= r1 < h:
        p = img.getpixel((c0, r0))
        if np.sum(p) > 0:
            pixels[c1, r1] = p
    return nxt


def apply_to(func, img, *, xlim=(-1, 1),  ylim=(-1, 1), lim=None, scale=None):
    if lim is not None:
        xlim = lim
        ylim = lim

    if scale is not None:
        img = img.resize((scale * img.width, scale * img.height), Image.NEAREST)

    w, h = img.size
    newImg = Image.new(img.mode, (w, h))
    pixels = newImg.load()

    x = np.linspace(xlim[0], xlim[1], w)
    y = np.linspace(ylim[0], ylim[1], h) * 1j
    z0 = np.sum(np.transpose([np.tile(x, h), np.repeat(y, w)]), axis=1)
    z1 = func(z0)
    reduce(
        partial(_apply_func_to_img, img, w, h, xlim, ylim, pixels),
        enumerate(z1)
    )

    return newImg
