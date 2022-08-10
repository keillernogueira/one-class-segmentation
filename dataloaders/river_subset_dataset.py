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

    imgs = ['no', 'so', 'se']

    for img_name in imgs:
        img = imageio.imread(os.path.join(args.dataset_path, img_name + '_image.tif'))
        mask = imageio.imread(os.path.join(args.dataset_path, img_name + '_mask.tif'))
        print(img.shape, mask.shape)

        if img_name == 'no':  # 13361 x 14318
            y = 3433  # trocado por imageio le trocado
            x = 11031
        elif img_name == 'so':  # 13350 x 14318
            y = 425
            x = 3281
        elif img_name == 'se':  # 13362 x 14329
            y = 5449
            x = 2421

        print(np.bincount(mask[x:x+1000, y:y+1000].flatten().astype(int)))
        # imageio.imwrite(os.path.join(args.dataset_path, img_name + '_image_crop.tif'), img[x:x+1000, y:y+1000, :])
        # imageio.imwrite(os.path.join(args.dataset_path, img_name + '_mask_crop.tif'), mask[x:x+1000, y:y+1000])
