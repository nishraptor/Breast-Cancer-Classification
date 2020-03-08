import numpy as np
import matplotlib.pyplot as plt
from transferlearningmodel import Net
from DataLoader import Dataset
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torchvision
import torchvision.transforms as transform

# Test dataset using pytorch's CIFAR10 class which contains classes for ‘airplane’,
# ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in
# CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.
# transform = transform.Compose([trasnforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)
# testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)


# train Function
# description: train a neural network model on the training set and test the validation set
# args: train_loader - tuples of images and labels for the training set
#       val_loader - tuples of images and labels for the validation set
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
    criterion = nn.BCELost()

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
        # this for loops is looking for a tuple(?) of images and labels
        for (imgs, labels) in train_loader:

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
        for (imgs, labels) in val_loader:

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

# get_loader function
# description: takes a csv file with information of the image dataset and splits the dataset
#              into testing, training, and validation set
# args: dataset - csv file name
#       test_size - proportion of dataset (between 0 and 1) to use as test test
#       train_size - proportion of dataset (between 0 and 1) to use as training set
#       val_size - proportion of dataset (between 0 and 1) to use as validation set
#       random_state - seed used by random number generator
#       shuffle - whether or not to shuffle the data before splitting
#       stratify - if not None, data is split in a stratified fashion
def get_loader(dataset, test_size, train_size, val_size, random_state, shuffle, stratify):

    # have dataloader class read in the csv file
    samples = DataLoader(dataset)

    # create list of image matrices
    matrixes = []
    matrixes = [matrixes.append(x['image']) for x in samples]

    # create list of image classification (either malignant or benign)
    targets = []
    matrixes = [matrixes.append(x['label']) for x in samples]

    # split the dataset into training/validation and testing lists
    X_train, X_test, y_train, y_test = train_test_split(matrixes, target, test_size = test_size, train_size = train_size + val_size, random_state = random_state, shuffle = shuffle, stratify = stratify)

    # split the training/validation list further into training and validation lists
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_size, train_size = train_size, random_state = random_state, shuffle = shuffle, stratify = stratify)

    # concatenate the images and labels into a tuple for each set type
    train = [(X_train[i], y_train[i]) for i in range(0, len(X_train))]
    val = [(X_val[i], y_val[i]) for i in range(0, len(X_val))]
    test = [(X_test[i], y_test[i]) for i in range(0, len(X_test))]

    return train, val, test


def evaluate(model, test_loader):
    # evaluation function at the end
    pass

def generatePlots():
    # plot the loss wrt iteration/epoch
    pass


def main():

    # create train loader, test loader, and validation loader

    # here I (Huy) created another csv that had the correct path to my image data
    # called test.csv, make sure to change the path to the csv as well as the
    # full_path column in the github csv
    train_loader, val_loader, test_loader = get_loader("../test.csv")

    # train the loader
    train(train_loader, val_loader)
