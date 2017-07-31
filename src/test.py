import sys
import transform
import math

from PIL import Image, ImageFilter
from transform import warp

if __name__ == '__main__':
    img = Image.open(f'img/{sys.argv[1]}.png')
    outName = 'out.png'
    if len(sys.argv) > 2:
        scale = float(sys.argv[2])
        img = img.resize(
            (int(scale * img.width), int(scale * img.height)),
            Image.NEAREST
        )
    if len(sys.argv) > 3:
        outName = f'{sys.argv[3]}.png'

    wimg = warp(
        lambda z:  math.e**(z**2),
        img,
        lim_in=(-2, 2),
        xlim_out=(-1, 3),
        ylim_out=(-2, 2),
        blend_func=transform.blend_add
    )
    wimg.show()
    wimg.save(outName)
