import os
import numpy as np
import cv2
from argparse import ArgumentParser
from libs.util import ImageChunker


def parse_args():
    parser = ArgumentParser(description='Unify all images in dataset with size (512, 512) by chunking')

    parser.add_argument(
        '-input_path', '--input_path',
        type=str, default='/root/CelebA/Img/img_align_celeba_png/',
        help='dataset path with images in different size'
    )

    parser.add_argument(
        '-out_path', '--out_path',
        type=str, default='/root/CelebA/Img/img_align_celeba_png_512/',
        help='dataset path with images in unified size'
    )

    parser.add_argument(
        '-size', '--size',
        type=str, default='512x512',
        help='target size of output image'
    )

    parser.add_argument(
        '-overlap', '--overlap',
        type=int, default=30,
        help='overlap when chunking image'
    )

    parser.add_argument(
        '-sid', '--sid',
        type=int, default=None,
        help='start id of image list'
    )

    parser.add_argument(
        '-eid', '--eid',
        type=int, default=None,
        help='end id of image list'
    )

    return parser.parse_args()

# Run script
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()
    # get input and output path for image processing
    input_path = args.input_path
    out_path = args.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # get target image size
    tar_rows, tar_cols = np.array(args.size.split('x'), dtype=np.int32)
    chunker = ImageChunker(tar_rows, tar_cols, args.overlap)

    # get image list from input dataset
    img_list = os.listdir(input_path)
    for i, img_name in enumerate(img_list):
        if i < args.sid:
            continue
        if i >= args.eid:
            break
        if i % 1000 == 0:
            print('processing {:d}/{:d} images'.format(i, len(img_list)))
        ori_img = cv2.imread(os.path.join(input_path, img_name))
        rimg = cv2.resize(ori_img, (420, 512))
        # if ori_img.shape[0] != 218 or ori_img.shape[1] != 178:
        #     print('{} with shape {:d}x{:d}'.format(os.path.join(input_path, img_name), ori_img.shape[0], ori_img.shape[1]))
        chunked_images = chunker.dimension_preprocess(rimg)
        for j in range(len(chunked_images)):
            if j == 0:
                cv2.imwrite(os.path.join(out_path, img_name), chunked_images[j])
            else:
                cv2.imwrite(os.path.join(out_path, img_name[:-4]+'_'+str(j)+'.png'), chunked_images[j])

