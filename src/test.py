import sys
import transform

from PIL import Image
from transform import apply_to

if __name__ == '__main__':
    img = Image.open(f'img/{sys.argv[1]}.png')
    if len(sys.argv) > 2:
        scale = float(sys.argv[2])
        img = img.resize(
            (int(scale * img.width), int(scale * img.height)),
            Image.NEAREST
        )
    apply_to(
        lambda z: z**2.5,
        img,
        lim_in=(-2, 2),
        blend_func=transform.blend_avg
    ).show()
