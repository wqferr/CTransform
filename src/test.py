from PIL import Image
from transform import apply_to
import sys

if __name__ == '__main__':
    img = Image.open(f'img/{sys.argv[1]}.png')
    if len(sys.argv) > 2:
        scale = float(sys.argv[2])
        img = img.resize(
            (int(scale * img.width), int(scale * img.height)),
            Image.NEAREST
        )
    apply_to(
        lambda z: z**5,
        img,
        lim_out=(-0.1, 0.1),
        dim_out=(1000, 1000)
    ).show()
