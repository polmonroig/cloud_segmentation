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
N_CLASSES = 5 # 4 + 1 empty class
CLASSES = ["Fish", "Flower", "Gravel", "Sugar"]


# FUNCTIONS
def rle2mask(rle, shape):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[0], shape[1]), order='F')


def mask2rle(image):
    pixels = image.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def make_masks(labels, image_name, shape):
    encoded_masks = labels.loc[labels['Image_Label'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle2mask(label, shape)
            masks[:, :, idx] = mask

    return masks
