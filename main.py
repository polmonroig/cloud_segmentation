import wandb
import torch
from segNet import SegNet
from image_dataset import ImageDataset, get_train_val_set
import utils


def main():

    x_train, y_train, x_val, y_val = get_train_val_set(utils.DATA_DIR, utils.TRAIN_LABELS)
    print(len(x_val.drop_duplicates()))
    print(len(x_train.drop_duplicates()))
    print(len(x_val))
    print(len(x_train))






if __name__ == "__main__":
    main()