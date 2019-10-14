import wandb
import torch
from segNet import SegNet
from image_dataset import ImageDataset, get_train_val_set
import utils


def main():

    x_train, y_train, x_val, y_val = get_train_val_set(utils.TRAIN_LABELS)
    transform = None
    shape = (1400, 2100, 3)
    train_dataset = ImageDataset(utils.TRAIN_IMAGES, x_train, y_train, transform , shape)
    val_dataset = ImageDataset(utils.TRAIN_IMAGES, x_val, y_val, transform, shape)

    model = SegNet()




if __name__ == "__main__":
    main()