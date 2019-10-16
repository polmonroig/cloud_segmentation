import torch
import torch.nn as nn
from torchvision.models import vgg19



class SegNet(nn.Module):

    def __init__(self):
        super(SegNet, self).__init__()

        # define encoder
        self.encoder = vgg19(pretrained=True).features

        for param in self.encoder.parameters():
            param.requires_grad = False

        # define decoder
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512,
                               kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512,
                               kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=4,
                               kernel_size=3, stride=1, padding=1)




    def train(self):
        self.conv1.train()

    def forward(self, x):
        # forward encoder and save indices
        indices = []
        sizes = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                layer.return_indices = True
                sizes.append(x.size())
                x, indx = layer(x)
                indices.append(indx)
            else:
                x = layer(x)
        # forward decoder
        # print("EncoderOutput:", x.size())
        x = self.conv1(x)
        x = self.pool1(self.relu1(x), indices[-1], sizes[-1])
        indices.pop(-1)
        sizes.pop(-1)
        # print("Pool1Output:", x.size())
        x = self.conv2(x)

        x = self.pool2(self.relu2(x), indices[-1], sizes[-1])
        indices.pop(-1)
        sizes.pop(-1)
        # print("Pool2Output:", x.size())
        x = self.conv3(x)
        x = self.pool3(self.relu2(x), indices[-1], sizes[-1])
        indices.pop(-1)
        sizes.pop(-1)
        # print("Pool3Output:", x.size())
        x = self.conv4(x)
        x = self.pool4(self.relu2(x), indices[-1], sizes[-1])
        indices.pop(-1)
        sizes.pop(-1)
        # print("Pool4Output:", x.size())
        x = self.conv5(x)
        x = self.pool5(self.relu2(x), indices[-1], sizes[-1])
        # print("Pool5Output:", x.size())
        x = self.conv6(x)
        # print("FinalSize:", x.size())
        return x


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