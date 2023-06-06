import numpy as np
import nibabel as nib
import skimage
import os
import glob
import tqdm
import re

pad = [32, 32, 32]


def resample(nih, **kwargs):
    scale = np.diagonal(nih.affine)[: 3]
    image = np.flip(nih.get_fdata(), np.flatnonzero(scale > 0))
    new_shape = (image.shape * np.abs(scale)).astype(int)
    image = skimage.transform.resize(image.astype(np.float64), new_shape, **kwargs)
    return image


def get_box_index(label):
    mask_index = np.array((label > 0).nonzero())
    max_index = (mask_index.max(axis=1) + pad).clip(max=label.shape)
    min_index = (mask_index.min(axis=1) - pad).clip(min=0)
    center = np.ceil((max_index + min_index) / 2).astype(int)
    length = np.ceil((max_index - min_index) / 2).max().astype(int)
    max_index = (center + length).clip(max=label.shape)
    min_index = (center - length).clip(min=0)
    return max_index, min_index


def get_image_box(image, box):
    max_index, min_index = box
    return image[
        min_index[0]: max_index[0],
        min_index[1]: max_index[1],
        min_index[2]: max_index[2],
    ]


def main():
    image_path = '../dataset/nih_data/NSCLC1/CT'
    label_path = '../dataset/nih_data/NSCLC1/Seg'
    to_path = './nih/nsclc1_test'
    os.makedirs(to_path, exist_ok=True)

    names = glob.glob(os.path.join(image_path, '*.gz'))
    names = [os.path.split(f)[1] for f in names]

    for img_name in tqdm.tqdm(names):
        label_name = re.sub(r"(?=.nii.gz)", "-1", img_name)

        image = nib.load(os.path.join(image_path, img_name))
        label = nib.load(os.path.join(label_path, label_name))

        image = resample(image)
        label = resample(label, anti_aliasing=False, order=0)

        label_box = get_box_index(label)

        image = get_image_box(image, label_box)
        label = get_image_box(label, label_box)

        path_prefix = os.path.join(to_path, img_name.split('.')[0])
        os.makedirs(path_prefix, exist_ok=True)

        np.save(os.path.join(path_prefix, 'img.npy'), image.astype(np.int16))
        np.save(os.path.join(path_prefix, 'label.npy'), label.astype(np.int8))
        np.save(os.path.join(path_prefix, 'merge.npy'),
                np.stack((image, label), axis=-1).astype(np.int16))


if __name__ == '__main__':
    main()
