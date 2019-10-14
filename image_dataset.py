from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class ImageDataset(Dataset):

    def __init__(self, root, transform, labels):
        super(ImageDataset, self).__init__()

    def __getitem__(self, item):
        return item

    def __len__(self):
        return 0


def get_train_val_set(data_dir, label_dir, val=0.2):
    encodings = pd.read_csv(label_dir)
    encodings['Image_Label'] = encodings['Image_Label'].apply(lambda x: x.split('_')[0])
    msk = np.random.rand(len(encodings)) < (1 - val)
    print(len(msk))
    print(len(encodings))
    train_x = encodings[msk]['Image_Label']
    print(len(train_x))
    print("dssd")
    train_y = encodings[msk]['EncodedPixels']
    val_x = encodings[~msk]['Image_Label']
    val_y = encodings[~msk]['EncodedPixels']
    return train_x, train_y, val_x, val_y

