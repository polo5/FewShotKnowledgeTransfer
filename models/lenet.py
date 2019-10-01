import torch
import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    """
    For convenience so we can add in in nn.Sequential
    instead of doing it manually in forward()
    """
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class LeNet32(nn.Module):
    """
    For SVHN/CIFAR experiments
    """

    def __init__(self, n_classes):
        super(LeNet32, self).__init__()
        self.n_classes = n_classes

        self.layers = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            View((-1, 16*5*5)),
            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, n_classes))

    def forward(self, x):
        activations = None
        return self.layers(x), activations

    def print_shape(self, x):
        """
        For debugging purposes
        :param image: input of shape image shape
        """
        act = x
        for layer in self.layers:
            act = layer(act)
            print(layer, '---->', act.shape)


if __name__ == '__main__':
    import random
    import sys
    from torchsummary import summary

    random.seed(1234)  # torch transforms use this seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    ### LENET32
    x = torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)
    true_labels = torch.tensor([[2.], [3], [1], [8], [4]], requires_grad=True)
    model = LeNet32(n_classes=10)
    output, act = model(x)
    print("\nOUTPUT SHAPE: ", output.shape)

    summary(model, input_size=(3,32,32))

