import torch
import torch.nn as nn
from torchvision.models import vgg19



class SegNet(nn.Module):

    def __init__(self):
        super(SegNet, self).__init__()

        self.encoder = vgg19(pretrained=True).features

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512,
                               kernel_size=3, stride=1, padding=1)

        # define network

    def train(self):
        self.conv1.eval()

    def forward(self, x):
        x = self.encoder(x)
        print(x.size())
        return self.conv1(x)


def train_step(model, data_loader, optimizer, device):
    model = model.train()
    criterion = nn.CrossEntropyLoss()
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        image, target = data
        image = image.to(device)
        target = target.to(device)
        out = model(image)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print("[" + str(i) + "/" + str(len(data_loader)))


def eval_step(model, data_loader_val, device):
    model = model.eval()
    criterion = nn.CrossEntropyLoss()
    for i, data in enumerate(data_loader_val):
        image, target = data
        image = image.to(device)
        target = target.to(device)
        out = model(image)
        loss = criterion(out, target)
        if i % 50 == 0:
            print("[" + str(i) + "/" + str(len(data_loader_val)))