import os
import sys
import argparse

import imageio
import numpy as np
from skimage.morphology import dilation, square, disk

import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = None


def resize_image(input_image_path, output_image_path, resize_ratio, quality):
    im = Image.open(input_image_path)
    print(im.size, type(im.size), im.size[0])
    im = im.resize((im.size[0] // resize_ratio, im.size[1] // resize_ratio), Image.ANTIALIAS)
    im.save(output_image_path, optimize=True, quality=quality)
    return im


def dilate_image(input_path, output_path, disk_size):
    img = imageio.imread(input_path)
    print('before dilation', img.shape, np.bincount(img.astype(int).flatten()))

    dil_out = dilation(img, disk(disk_size))
    print('after dilation', img.shape, np.bincount(dil_out.astype(int).flatten()))

    # imageio.imwrite(output_path, dil_out)
    imageio.imwrite(output_path, dil_out * 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='resize_images')
    parser.add_argument('--input_image_path', type=str, required=True, help='Input image path.')
    parser.add_argument('--output_image_path', type=str, required=True, help='Output image path')

    parser.add_argument('--disk_size', type=int, required=False, default=50, help='Disk size for dilation')
    parser.add_argument('--resize_ratio', type=int, required=False, default=4, help='Resize ratio')
    parser.add_argument('--quality', type=int, required=False, default=10, help='Jpeg quality')
    args = parser.parse_args()
    print(sys.argv[0], args)

    output_path_disk = args.output_image_path + '_disk' + str(args.disk_size) + '.png'
    output_path_final = args.output_image_path + '_disk' + str(args.disk_size) + '_resize' + \
                        str(args.resize_ratio) + '_quality' + str(args.quality) + '.jpg'

    dilate_image(args.input_image_path, output_path_disk, args.disk_size)
    resize_image(output_path_disk, output_path_final, args.resize_ratio, args.quality)
