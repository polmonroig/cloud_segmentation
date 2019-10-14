from PIL import Image
from torch.utils.data import Dataset
from os.path import join
import pandas as pd
import numpy as np
import itertools
import torch
import utils


class ImageDataset(Dataset):

    def __init__(self, root, x, y, transform, shape):
        super(ImageDataset, self).__init__()
        self.x = x
        self.y = y
        self.shape = shape
        self.root = root

    def __getitem__(self, item):
        # The nth label of the nth image is n * 4 -> 4 labels per image
        masks = self.get_masks(self.y[item * 4:item * 4 + 3])
        image = Image.open(join(self.root, self.x[item]))

        return item, masks 

    def __len__(self):
        return 0

    def get_masks(self, encoded_masks):
        masks = torch.zeros((self.shape[0], self.shape[1], 4), dtype=torch.float32)
        for idx, label in enumerate(encoded_masks.values):
            if label is not np.nan:
                mask = utils.rle2mask(label, self.shape)
                masks[:, :, idx] = mask

        return masks


def extend4(msk):
    return np.array(list(itertools.chain.from_iterable(itertools.repeat(x, 4) for x in msk)))


def get_train_val_set(label_dir, val=0.2):
    encodings = pd.read_csv(label_dir)
    encodings['Image_Label'] = encodings['Image_Label'].apply(lambda x: x.split('_')[0])
    msk = np.random.rand(int(len(encodings) / 4)) < (1 - val)
    msk = extend4(msk)

    train_x = encodings[msk]['Image_Label'].drop_duplicates()
    train_y = encodings[msk]['EncodedPixels']
    val_x = encodings[~msk]['Image_Label'].drop_duplicates()
    val_y = encodings[~msk]['EncodedPixels']
    return train_x, train_y, val_x, val_y

