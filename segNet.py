import torch
import torch.nn as nn
from torchvision.models import vgg19
from utils import show_image_with_masks
import wandb


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
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512,
                               kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=4,
                               kernel_size=3, stride=1, padding=1)
        self.logits = nn.Sigmoid()


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
        x = self.conv3(self.relu2(x))
        x = self.pool2(self.relu3(x), indices[-1], sizes[-1])
        indices.pop(-1)
        sizes.pop(-1)
        # print("Pool2Output:", x.size())
        x = self.conv4(x)
        x = self.pool3(self.relu4(x), indices[-1], sizes[-1])
        indices.pop(-1)
        sizes.pop(-1)
        # print("Pool3Output:", x.size())
        x = self.conv5(x)
        x = self.pool4(self.relu5(x), indices[-1], sizes[-1])
        indices.pop(-1)
        sizes.pop(-1)
        # print("Pool4Output:", x.size())
        x = self.conv6(x)
        x = self.pool5(self.relu6(x), indices[-1], sizes[-1])
        # print("Pool5Output:", x.size())
        x = self.conv7(x)
        x = self.conv8(self.relu7(x))

        # print("FinalSize:", x.size())
        return self.logits(x)


def dice_loss(pred, target):
    smooth = 1.
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def train_step(model, data_loader, optimizer, device):
    model.train()
    criterion = dice_loss
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        image, target = data
        image = image.to(device)
        target = target.to(device)
        out = model(image)
        loss_fish = criterion(out[:,0, :, :], target[:,0, :, :])
        loss_flower = criterion(out[:,1, :, :], target[:,1, :, :])
        loss_gravel = criterion(out[:,2, :, :], target[:,2, :, :])
        loss_sugar = criterion(out[:,3, :, :], target[:,3, :, :])
        loss_fish.backward(retain_graph=True)
        loss_flower.backward(retain_graph=True)
        loss_gravel.backward(retain_graph=True)
        loss_sugar.backward(retain_graph=True)
        optimizer.step()
        if i % 50 == 0:
            # show_image_with_masks(image[0], out[0])
            print("[" + str(i) + "/" + str(len(data_loader)))
            acc1 = accuracy_score(target[:,0, :, :], out[:,0, :, :])
            acc2 = accuracy_score(target[:,1, :, :], out[:,1, :, :])
            acc3 = accuracy_score(target[:,2, :, :], out[:, 2])
            acc4 = accuracy_score(target[:,3, :, :], out[:,3, :, :])
            loss_total = (loss_sugar.item() + loss_flower.item() + loss_gravel.item() + loss_fish.item()) / 4
            acc_total = (acc1 + acc2 + acc3 + acc4) / 4
            print("Fish Loss/Accuracy:", loss_fish.item(), "/", acc1)
            print("Flower Loss/Accuracy:", loss_flower.item(),"/", acc2)
            print("Gravel Loss/Accuracy:", loss_gravel.item(), "/", acc3)
            print("Sugar Loss/Accuracy:", loss_sugar.item(),"/", acc4)
            print("Total Loss/Accuracy:", loss_total, "/", acc_total)
            # log data into wandb
            wandb.log({"loss_fish": loss_fish.item(), "accuracy_fish": acc1,
                       "loss_flower": loss_flower.item(), "accuracy_flower": acc2,
                       "loss_gravel": loss_gravel.item(), "accuracy_gravel": acc3,
                       "loss_sugar": loss_sugar.item(), "accuracy_sugar": acc4,
                       "loss_sugar": loss_sugar.item(), "accuracy_sugar": acc4,
                       "loss_total": loss_total, "accuracy_total":acc_total})



def accuracy_score(y_true, y_pred):
    total = 1
    for a in y_pred.size():
        total *= a
    return ((y_true.eq(y_pred.long())).sum() / float(total)).item()


def eval_step(model, data_loader_val, device):
    model.eval()
    criterion = dice_loss
    for i, data in enumerate(data_loader_val):
        image, target = data
        image = image.to(device)
        target = target.to(device)
        out = model(image)
        loss_fish = criterion(out[:, 0, :, :], target[:, 0, :, :])
        loss_flower = criterion(out[:, 1, :, :], target[:, 1, :, :])
        loss_gravel = criterion(out[:, 2, :, :], target[:, 2, :, :])
        loss_sugar = criterion(out[:, 3, :, :], target[:, 3, :, :])
        if i % 50 == 0:
            print("[" + str(i) + "/" + str(len(data_loader_val)))
            acc1 = accuracy_score(target[:, 0, :, :], out[:, 0, :, :])
            acc2 = accuracy_score(target[:, 1, :, :], out[:, 1, :, :])
            acc3 = accuracy_score(target[:, 2, :, :], out[:, 2])
            acc4 = accuracy_score(target[:, 3, :, :], out[:, 3, :, :])
            loss_total = (loss_sugar.item() + loss_flower.item() + loss_gravel.item() + loss_fish.item()) / 4
            acc_total = (acc1 + acc2 + acc3 + acc4) / 4
            print("Fish Loss/Accuracy:", loss_fish.item(), "/", acc1)
            print("Flower Loss/Accuracy:", loss_flower.item(), "/", acc2)
            print("Gravel Loss/Accuracy:", loss_gravel.item(), "/", acc3)
            print("Sugar Loss/Accuracy:", loss_sugar.item(), "/", acc4)
            print("Total Loss/Accuracy:", loss_total, "/", acc_total)
            # log data into wandb
            wandb.log({"val_loss_fish": loss_fish.item(), "val_accuracy_fish":acc1,
                       "val_loss_flower": loss_flower.item(), "val_accuracy_flower":acc2,
                       "val_loss_gravel": loss_gravel.item(), "val_accuracy_gravel":acc3,
                       "val_loss_sugar": loss_sugar.item(), "val_accuracy_sugar":acc4,
                       "val_loss_sugar": loss_sugar.item(), "val_accuracy_sugar":acc4,
                       "val_loss_total": loss_total, "val_accuracy_total":acc_total})
            show_image_with_masks(image[0], out[0])

