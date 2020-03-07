import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    # Constructor
    def __init__(self):
        super(Net, self).__init__()

        # Define private variable with pytorch's pretrained resnet50 model
        self.resnet50 = models.resnet50(pretrained = True)

        # Prevent layers other than the most recent layer from being trained
        # by removing the ability to accumulate gradients
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # Replace the last layer with our desired output size
        self.resnet50.fc = nn.Linear(2048, 1)

    # Forward function
    def forward(self, x):
        x = self.resnet50(x)
        return x




net = Net()
print(net)
