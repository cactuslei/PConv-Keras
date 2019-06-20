# Split the CelebA dataset into training/validation/testing.
# The identities in training, validation and testing should be disjoint.
# There are total 202599 images and the validation and testing set are both set to 3K.
# The remaining 142599 images are used as training set.
import os
from argparse import ArgumentParser
import numpy as np

def parse_args():
    parser = ArgumentParser(description='Split CelebA into training(142,599), validation(3K) and testing(3K) set.')

    parser.add_argument(
        '-data_path', '--data_path',
        type=str, default='/userhome/CelebA/Img/img_align_celeba_png/',
        help='Dataset with all images'
    )

    parser.add_argument(
        '-val_path', '--val_path',
        type=str, default='/userhome/CelebA/Img/img_align_celeba_png_val/',
        help='Where to output validation images'
    )

    parser.add_argument(
        '-test_path', '--test_path',
        type=str, default='/userhome/CelebA/Img/img_align_celeba_png_test/',
        help='Where to output validation images'
    )

    parser.add_argument(
        '-id_file', '--id_file',
        type=str, default='/userhome/CelebA/Anno/identity_CelebA.txt',
        help='identities of the images'
    )

    return parser.parse_args()

# Run script
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    data_path = args.data_path
    val_path = args.val_path
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    test_path = args.test_path
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    # get identities of the images
    with open(args.id_file) as f:
        lines = f.readlines()
    maxid = -1
    minid = 202599
    id2img = dict()
    for line in lines:
        cid = int(line.split(' ')[1])
        if cid in id2img:
            id2img[cid].append(line.split(' ')[0].replace('jpg', 'png'))
        else:
            id2img[cid] = []
            id2img[cid].append(line.split(' ')[0].replace('jpg', 'png'))
        if cid > maxid:
            maxid = cid
        if cid < minid:
            minid = cid
    print('maxid: {:d}, minid: {:d}'.format(maxid, minid))
    rand_list = np.arange(minid, maxid+1)
    np.random.seed(42)
    np.random.shuffle(rand_list)
    # select identities for testing set
    total_imgs = 0
    sid = 0
    for id in rand_list:
        for img_name in id2img[id]:
            os.rename(os.path.join(data_path, img_name), os.path.join(test_path, img_name))
            total_imgs += 1
        sid += 1
        if total_imgs > 3000:
            break
    # select identities for validation set
    total_imgs = 0
    for id in rand_list[sid:]:
        for img_name in id2img[id]:
            os.rename(os.path.join(data_path, img_name), os.path.join(val_path, img_name))
            total_imgs += 1
        if total_imgs > 3000:
            break

    # rename data_path folder with 'train' suffix
    if data_path[-1] == '/':
        data_path = data_path[:-1]
    os.rename(data_path, data_path+'_train')
    os.mkdir(data_path)
    os.mkdir(os.path.join(data_path, 'train'))
    os.rename(data_path + '_train', os.path.join(data_path, 'train', 'imgs'))
    os.mkdir(os.path.join(data_path, 'val'))
    os.rename(val_path, os.path.join(data_path, 'val', 'val'))
    os.mkdir(os.path.join(data_path, 'test'))
    os.rename(test_path, os.path.join(data_path, 'test', 'imgs'))



