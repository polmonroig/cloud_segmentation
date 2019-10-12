"""utils.py: this file saves some global
                 variables that need to be shared accross files
                 and some utility functions
"""
from os.path import join
import numpy as np


# VARIABLES
MODEL_DIR = "models/"
DATA_DIR = "data/understanding_cloud_organization/"
TRAIN_IMAGES = join(DATA_DIR, "train_images/")
TRAIN_LABELS = join(DATA_DIR, "train.csv")
TEST_IMAGES = join(DATA_DIR, "test_images/")
N_CLASSES = 4


# FUNCTIONS
def rle2mask(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def mask2rle(image):
    pixels= image.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
