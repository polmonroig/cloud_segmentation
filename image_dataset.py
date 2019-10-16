from PIL import Image
from cv2 import resize
from torch.utils.data import Dataset
from os.path import join
import pandas as pd
import numpy as np
import itertools
import torch
import utils


class ImageDataset(Dataset):

    def __init__(self, root, x, y, transforms, shape):
        super(ImageDataset, self).__init__()
        self.x = x
        self.y = y
        self.shape = shape
        self.root = root
        self.transforms = transforms

    def __getitem__(self, item):
        # The nth label of the nth image is n * 4 -> 4 labels per image
        masks = self.get_masks(self.y[item * 4:item * 4 + 3])
        image = Image.open(join(self.root, self.x[item]))

        if self.transforms is not None:
            image = self.transforms(image)

        return image, masks

    def __len__(self):
        return len(self.x)

    def get_masks(self, encoded_masks):
        masks = np.zeros((self.shape[0], self.shape[1], 4), dtype=np.float32)
        for idx, label in enumerate(encoded_masks.values):
            if label is not np.nan:
                mask = utils.rle2mask(label, self.shape)
                masks[:, :, idx] = mask
        resized_masks = np.zeros((350, 525, 4), dtype=np.float32)
        for idx in range(4):
            resized_masks[:, :, idx] = resize(masks[:, :, idx], (525, 350))

        return torch.as_tensor(resized_masks, dtype=torch.float32)


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
    return np.array(train_x.values), train_y, \
           np.array(val_x.values), val_y.values

