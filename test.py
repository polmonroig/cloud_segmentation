from segNet import  SegNet
from image_dataset import  ImageDataset
import utils
import os
from PIL import Image
import torch
import csv


model = SegNet()
model_dir = "wandb/run-20191017_073956-zukd8wh5/model.pt"
model.load_state_dict(torch.load(model_dir))
model.eval()

transforms = utils.get_transforms()
encodes = [["Image_Label", "EncodedPixels"]]
for path in sorted(os.listdir(utils.TEST_IMAGES)):
    image = Image.open(os.path.join(utils.TEST_IMAGES, path))
    image = transforms(image)
    out = model(image)
    for cat, mask in zip(out, utils.CLASSES):
        current_name = path + "_" + cat
        encodes.append([current_name, utils.mask2rle(mask)])

with open("submission.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for row in encodes:
        wr.writerow(row)