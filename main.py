import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
import os
from torch.utils import data

import numpy as np
import matplotlib.pyplot as plt
from transferlearningmodel import Net
from DataLoader import Dataset

import torch
from torch import nn
import torchvision
import torchvision.transforms as transform

from sklearn.model_selection import train_test_split

import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import glob
import time

import os.path

import matplotlib.pyplot as plt
from transferlearningmodel import Net
from DataLoader import Dataset

from torch.utils.data.sampler import SubsetRandomSampler

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
def train_model(train_loader, val_loader):

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
    criterion = nn.BCELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    best_val_loss = float('inf') 
    # hold off on variable for now (use this to check if the model loss is increasing x number of times)

    # Runs through one batch in the train_loader
    for epoch in range(1):
        print("epoch: ", epoch)
        # Define variables to accumulate the loss over the total epoch
        train_epoch_loss_var = 0
        val_epoch_loss_var = 0

        # This for loop is for the training set data
        # ********** Coordinate with the dataloader *************
        # this for loops is looking for a tuple(?) of images and labels
#         for imgs, (_, labels) in train_loader:
        for i, (imgs,labels) in enumerate(train_loader):
        
            print("minibatch: ", i, " out of ", len(train_loader))
            
            labels = labels.view(labels.size(0), 1)
            # Zeroes the gradients so they don't accumulate
            optimizer.zero_grad()

            # Pass one image forward through the cnn
            outputs = model(imgs)

            # Assigns the loss function to the outputs
            loss = criterion(outputs, labels.float())
#             clear(outputs)

            # Computes the gradients with respect to the cost function
            loss.backward()

            # Move in the direction of gradient descent
            optimizer.step()

            train_loss_per_iter.append(loss)
            train_epoch_loss_var += loss
            
            print("loss: ", loss)


        # This for loop is for the validation set data
        for  i, (imgs,labels) in enumerate(val_loader):
            
            labels = labels.view(labels.size(0), 1)
            # Pass one image forward through the cnn
            outputs = model(imgs)

            # Assigns the loss function to the outputs
            loss = criterion(outputs, labels.float())

            # Save loss per iteration
            val_loss_per_iter.append(loss)
            val_epoch_loss_var += loss

    # Save values of loss over total epoch
    train_epoch_loss.append(train_epoch_loss_var)
    val_epoch_loss.append(val_epoch_loss_var)
    
    torch.save(model, 'saved_model.pt')
    
    return  train_loss_per_iter, train_epoch_loss, val_loss_per_iter, val_epoch_loss, val_epoch_acc

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
batch_size = 2

def loadData():
 
    #Create the dataset
    dataset = Dataset(csv_file = 'csv.csv', \
                     transform = transform.Compose([transform.Resize((100,100)), \
                                                    transform.ToTensor(),\
                                                   transform.Normalize((0.5, 0.5, 0.5), \
                                                                        (0.5, 0.5, 0.5))]))
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split1 = int(np.floor(.6 * dataset_size))
    split2 = int(np.floor(.8 * dataset_size))
    train_indices, val_indices, test_indices = indices[: split1], indices[split1: split2], indices[split2: ]
 
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
 
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    valloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=valid_sampler)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=test_sampler)
 
 
    return trainloader, valloader, testloader

def show_batch(images, labels):
 
        batch_size = len(images)
        im_size = images.size(0)
        grid_border_size = 3
     
        grid = make_grid(images)
        plt.imshow(grid.numpy().transpose())
        plt.title('Batch from dataloader')
        plt.show()

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
