"""Convolutional Autoencoder"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path
import time
from contextlib import contextmanager
from typing import Callable, Generator

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class CAE(nn.Module):
    def __init__(self): 
        super(CAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.t_conv1 = nn.ConvTranspose2d(4, 16, kernel_size=2, stride=2, padding=0)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, padding=0)
        
    def encode(self, x):

        x = F.relu(self.conv1(x)) 
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x) 

        return x 

    def decode(self, z):
        z = F.relu(self.t_conv1(z))
        z = torch.sigmoid(self.t_conv2(z)) 
        return z 

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
   
def train(model, device, train_loader, optimizer, criterion):
    model.train()

    train_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data) 
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
     
    train_loss /= len(train_loader.dataset) 
    return train_loss 

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad(): 
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data) 
            loss = criterion(output, data) 
            test_loss += loss.item() 

    test_loss /= len(test_loader.dataset)
    return test_loss


def train_epoch(args, model, device, train_loader, test_loader, optimizer, criterion):
    """Main training function"""
    losses = [] # store training loss and test loss
    min_test_loss = np.inf # initialize min test loss to be infity
    best_epoch = None # record which epoch has smallest test loss

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        test_loss = test(model, device, test_loader, criterion)
        losses.append([train_loss, test_loss])

        # Give status reports every 5 epochs
        if (epoch % 5 == 0) or epoch == 1:
            print(f" EPOCH {epoch}. Progress: {epoch/(args.epochs+1)*100}%. ") 
            print(f" Train loss: {train_loss:.6f}. Test loss: {test_loss:.6f}")

        if test_loss < min_test_loss:
            min_test_loss = test_loss 
            best_epoch = epoch 

        print("Saving checkpoint...")
        torch.save(model.state_dict(), "./checkpoint.pt")
                
    print(f"Min test loss of {min_test_loss:.3f} occurred after epoch {best_epoch}.")

    with open(f'./loss.npy', 'wb') as f: 
        np.save(f, np.array(losses))


def main():
    #=============================
    # hyperparameter from command line 
    #=============================
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
   
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    #=============================
    # load data
    #=============================
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    #=============================
    # initialize model, optimizer, loss function
    #=============================
    model = CAE().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.MSELoss()

    #=============================
    # training
    #=============================
    train_epoch(args, model, device, train_loader, test_loader, optimizer, criterion)

    #=============================
    # evaluation
    #=============================


if __name__ == '__main__':
    main()






