from PIL import Image
import numpy as np


def apply_to(func, img, *, xlim=(-1, 1),  ylim=(-1, 1), lim=None, scale=None):
    if lim is not None:
        xlim = lim
        ylim = lim

    if scale is not None:
        img = img.resize((scale * img.width, scale * img.height))

    w, h = img.size
    newImg = Image.new('RGB', (w, h))
    pixels = newImg.load()

    x = np.linspace(xlim[0], xlim[1], w)
    y = np.linspace(ylim[0], ylim[1], h)
    z0 = np.array([a + b * 1j for a in x for b in y]).reshape(w, h)
    z1 = func(z0)

    rmax, rmin = 0, 0
    cmax, cmin = 0, 0
    for c0, zs in enumerate(z1):
        for r0, z in enumerate(zs):
            c1 = round(w * (np.real(z) - xlim[0]) / (xlim[1] - xlim[0]))
            r1 = round(h * (np.imag(z) - ylim[0]) / (ylim[1] - ylim[0]))
            if 0 <= c1 < w and 0 <= r1 < h:
                p = img.getpixel((c0, r0))
                if np.sum(p) > 0:
                    pixels[c1, r1] = p

    return newImg
