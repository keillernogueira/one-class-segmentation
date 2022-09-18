import os
import argparse
import imageio
import numpy as np

import PIL
from PIL import Image, ImageDraw
PIL.Image.MAX_IMAGE_PIXELS = 933120000


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path.')
    args = parser.parse_args()
    print(args)

    points = [(8442, 1945), (8378, 3305), (5378, 4765)]

    img = imageio.imread(os.path.join(args.dataset_path, 'image.tif'))
    mask = imageio.imread(os.path.join(args.dataset_path, 'mascara.tif'))
    print(img.shape, mask.shape)

    for i, p in enumerate(points):
        x = p[1]
        y = p[0]

        print(np.bincount(mask[x:x+1000, y:y+1000].flatten().astype(int)))
        imageio.imwrite(os.path.join(args.dataset_path, 'crop_' + str(i) + '.tif'), img[x:x+1000, y:y+1000, :])
        imageio.imwrite(os.path.join(args.dataset_path, 'mask_' + str(i) + '.tif'), mask[x:x+1000, y:y+1000])
