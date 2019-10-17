import wandb
import torch
from segNet import SegNet, train_step, eval_step
from image_dataset import ImageDataset, get_train_val_set
import torch.optim as optim
import utils


def main():
    # CHANGE LOSS FUNCTION TO CORRECT ONE
    wandb.init(project="cloud_segmentation")
    # Setup device selection
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print("Running on gpu")
    else:
        print("Running on cpu")
    # define hyper-paremeters
    batch_size = 2
    learning_rate = 0.001
    n_epochs = 2
    wandb.config.update({"epochs": n_epochs, "batch_size": batch_size, "learning_rate": learning_rate})

    # Setup image transforms and data augmentation
    transforms = utils.get_transforms(False)

    # split train test set
    x_train, y_train, x_val, y_val = get_train_val_set(utils.TRAIN_LABELS)
    shape = (1400, 2100, 3)
    train_dataset = ImageDataset(utils.TRAIN_IMAGES, x_train, y_train, transforms , shape)
    val_dataset = ImageDataset(utils.TRAIN_IMAGES, x_val, y_val, transforms, shape)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define and train model
    model = SegNet()
    wandb.watch(model, log="all")
    model.to(device)

    # inputs, classes = next(iter(data_loader_val))
    # out = model(inputs.to(device))
    # utils.show_image_with_masks(inputs[0], out[0])

    optimizer = optim.Adam(model.parameters(), learning_rate)

    # torch.save(model.state_dict(), wandb.run.dir + "/model.pt")
    for epoch in range(n_epochs):
        print("Epoch:", epoch)
        train_step(model, data_loader, optimizer, device)
        eval_step(model, data_loader_val, device)

    # save model to W&B
    torch.save(model.state_dict(), wandb.run.dir + "/model.pt")

if __name__ == "__main__":
    main()
