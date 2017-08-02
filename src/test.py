import sys
import transform
import math

from PIL import Image
from transform import warp
from numpy import vectorize

if __name__ == '__main__':
    img = Image.open(f'img/{sys.argv[1]}.png')
    outName = 'out.png'
    dim_out = img.size
    if len(sys.argv) > 2:
        detail = float(sys.argv[2])
        img = img.resize(
            (int(detail * img.width), int(detail * img.height)),
            Image.NEAREST
        )
    if len(sys.argv) > 3:
        outName = f'{sys.argv[3]}.png'

    wimg = warp(
        vectorize(lambda z: math.e ** (z * z)),
        img,
        lim_in=(-2, 2),
        lim_out=(-4, 4),
        dim_out=dim_out,
        blend_func=transform.blend_avg
    )
    wimg.show()
    wimg.save(outName)
