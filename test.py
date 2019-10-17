from segNet import SegNet
from image_dataset import ImageDataset
import utils
import os
from PIL import Image
import torch
import csv
import matplotlib.pyplot as plt


device = torch.device('cuda')
model = SegNet()
model_dir = "wandb/run-20191017_073956-zukd8wh5/model.pt"
model.load_state_dict(torch.load(model_dir))
model.eval()
model = model.to(device)

transforms = utils.get_transforms(False)
shape = (1400, 2100, 3)
test_dataset = ImageDataset(utils.TEST_IMAGES, os.listdir(utils.TEST_IMAGES), None, transforms, shape, True)
batch_size = 1
data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

encodes = [["Image_Label", "EncodedPixels"]]
for i, data in enumerate(data_loader):
    image, path = data
    image = image.to(device)
    out = model(image.view(-1, 3, 350, 525))
    out = out.cpu().detach().numpy()
    print(str(i) + "/" + str(len(os.listdir(utils.TEST_IMAGES))))
    plt.imshow(utils.conv_image(image[0]))
    plt.show()
    for mask, cat in zip(out[0], utils.CLASSES):
        current_name = path[0] + "_" + cat
        mask, n_masks = utils.post_process(mask, 0.65 , 10000)
        if n_masks != 0:
            encodes.append([current_name, utils.mask2rle(mask)])
        else:
            encodes.append([current_name, ""])



with open("submission.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for row in encodes:
        wr.writerow(row)


