import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.optim as optim


# source: https://medium.com/dair-ai/building-rnns-is-fun-with-pytorch-and-google-colab-3903ea9a3a79


BATCH_SIZE = 64

# list all transformations
transform = transforms.Compose(
    [transforms.ToTensor()])

# download and load training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

# download and load testing dataset
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)


# functions to show an image
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))

N_STEPS = 28
N_INPUTS = 28
N_NEURONS = 150
N_OUTPUTS = 10
N_EPHOCS = 20



class BasicRNN(nn.Module):
    def __init__(self, n_steps, n_inputs, n_neurons, n_outputs):
        super(BasicRNN, self).__init__()

        self.Wx = torch.nn.Parameter(torch.randn(n_inputs, n_neurons))  # n_inputs X n_neurons
        self.Wx.requires_grad = True

        self.Wy = torch.nn.Parameter(torch.randn(n_neurons, n_neurons))  # n_neurons X n_neurons
        self.Wy.requires_grad = True

        self.b = torch.nn.Parameter(torch.zeros(1, n_neurons))  # 1 X n_neurons
        self.b.requires_grad = True

        self.n_steps = n_steps
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs

        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    def forward(self, X, h):
        # transforms X to dimensions: n_steps X batch_size X n_inputs = 28 x 64 x 28
        X = X.permute(1, 0, 2)
        batch_size = X.size(1)
        h = torch.zeros(batch_size, self.n_neurons)
        for i in range(self.n_steps):
            h = torch.tanh(torch.mm(h, self.Wy) +
                                 torch.mm(X[i, :, :], self.Wx) + self.b)  # batch_size X n_neurons
            out = self.FC(h)
            #outputs.append(out.view(-1, self.n_outputs)) # batch_size X n_output
        return out, h


# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model instance
model = BasicRNN(N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def get_accuracy(logit, target, batch_size):
    """ Obtain accuracy for training round """
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()


for epoch in range(N_EPHOCS):  # loop over the dataset multiple times
    train_running_loss = 0.0
    train_acc = 0.0
    model.train()

    # TRAINING ROUND
    for i, data in enumerate(trainloader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # still trains well without init probably because it's done in forward pass
        # reset hidden states
        hidden = model.init_hidden()

        # get the inputs
        inputs, labels = data
        inputs = inputs.view(-1, 28, 28)

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(outputs, labels, BATCH_SIZE)

    model.eval()
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f'
          % (epoch, train_running_loss / i, train_acc / i))

