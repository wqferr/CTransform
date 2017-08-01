import numpy as np

from PIL import Image
from functools import partial
from numpy import vectorize


def blend_first(p):
    return p[0]


def blend_add(p):
    a = np.array(p)
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    return (
        min(int(round(np.sum(r))), 255),
        min(int(round(np.sum(g))), 255),
        min(int(round(np.sum(b))), 255)
    )


def blend_avg(p):
    a = np.array(p)
    r, g, b = a[:, 0], a[:, 1], a[:, 2]

    return (
        int(round(np.average(r))),
        int(round(np.average(g))),
        int(round(np.average(b))),
    )


def filter_nonempty(p):
    if len(p) > 3 and p[3] == 0:
        return False
    return np.sum(p[:3]) > 0


def _apply_func_to_img(filter_func,
                       dim_in, dim_out,
                       xlim_in, ylim_in,
                       xlim_out, ylim_out,
                       in_pixels, out_pixels,
                       z0, z1):
    c0 = int((dim_in[0] - 1) * (np.real(z0) - xlim_in[0]) /
             (xlim_in[1] - xlim_in[0]))
    r0 = int((dim_in[1] - 1) * (np.imag(z0) - ylim_in[0]) /
             (ylim_in[1] - ylim_in[0]))

    c1 = int(round((dim_out[0] - 1) * (np.real(z1) - xlim_out[0]) /
                   (xlim_out[1] - xlim_out[0])))
    r1 = int(round((dim_out[1] - 1) * (np.imag(z1) - ylim_out[0]) /
                   (ylim_out[1] - ylim_out[0])))
    if 0 <= c1 < dim_out[0] and 0 <= r1 < dim_out[1]:
        p = in_pixels[r0][c0]
        if filter_func(p):
            out_pixels[c1][r1].append(p)


def warp(func, img, *,
         xlim_in=(-1, 1), xlim_out=(-1, 1),
         ylim_in=(-1, 1), ylim_out=(-1, 1),
         lim_in=None, lim_out=None,
         dim_out=None,
         blend_func=blend_first,
         filter_func=filter_nonempty):
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
    in_pixels = np.array(img)

    part = partial(
        _apply_func_to_img,
        filter_func,
        dim_in, dim_out,
        xlim_in, ylim_in,
        xlim_out, ylim_out,
        in_pixels, pixList
    )
    vectorize(part)(z0, z1)

    for i, r in enumerate(pixList):
        for j, p in enumerate(r):
            if p:
                pixels[i, j] = blend_func(p)
    return newImg
