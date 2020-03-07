import numpy as np
import matplotlib.pyplot as plt
from transferlearningmodel import Net

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

# Test dataset using pytorch's CIFAR10 class which contains classes for ‘airplane’,
# ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in
# CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.
# transform = transform.Compose([trasnforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)
# testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

def train(train_loader, val_loader):

    # Define relevant testing parameters - the loss per iteration/epoch for training
    # and validation sets
    train_loss_per_iter = []
    train_epoch_loss = []
    val_loss_per_iter = []
    val_epoch_loss = []
    val_epoch_acc = []

    # Define the cnn model
    model = Net()

    # Define the cost function
    criterion = BCELost()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    best_val_loss = float('inf') # hold off on variable for now (use this to check if the model loss is increasing x number of times)

    # Runs through one batch in the train_loader
    for epoch in range(3):

        # Define variables to accumulate the loss over the total epoch
        train_epoch_loss_var = 0
        val_epoch_loss_var = 0

        # This for loop is for the training set data
        # ********** Coordinate with the dataloader *************
        # this for loops is looking for a tuple(?) of iamges and labels
        for (imgs, labels) in enumerate(train_loader, total=len(train_loader)):

            # Zeroes the gradients so they don't accumulate
            optimizer.zero_grad()

            # Pass one image forward through the cnn
            outputs = model(imgs)

            # Assigns the loss function to the outputs
            loss = criterion(outputs, labels)
            clear(outputs)

            # Computes the gradients with respect to the cost function
            loss.backward()

            # Move in the direction of gradient descent
            optimizer.step()

            train_loss_per_iter.append(loss)
            train_epoch_loss_var += loss


        # This for loop is for the validation set data
        for (imgs, labels) in enumerate(val_loader, total=len(val_loader)):

            # Pass one image forward through the cnn
            outputs = model(imgs)

            # Assigns the loss function to the outputs
            loss = criterion(outputs, labels)

            # Save loss per iteration
            val_loss_per_iter.append(loss)
            val_epoch_loss_var += loss

    # Save values of loss over total epoch
    train_epoch_loss.append(train_epoch_loss_var)
    val_epoch_loss.append(val_epoch_loss_var)


def evaluate(model, test_loader):
    # evaluation function at the end
    pass

def generatePlots():
    # plot the loss wrt iteration/epoch
    pass


def main():

    # create trainloader, testloader, and validationloader
    # maybe use prepare-loader (method Nishant made)

    pass
