from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, root, transform):
        super(ImageDataset, self).__init__()

    def __getitem__(self, item):
        return item

    def __len__(self):
        return 0