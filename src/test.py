from PIL import Image
from transform import apply_to
import sys

if __name__ == '__main__':
    apply_to(
        lambda z: z * z,
        Image.open(f'img/{sys.argv[1]}.png')
    ).show()
