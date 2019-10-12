import wandb
import torch
from segNet import SegNet
from image_dataset import ImageDataset
import utils

def main():

    transform = None
    dataset = ImageDataset(utils.DATA_DIR, transform)

    model = SegNet()







if __name__ == "__main__":
    main()