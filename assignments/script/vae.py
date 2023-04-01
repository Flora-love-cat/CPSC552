"""
Variational Autoencoder 

!python vae.py --batch-size 128 --epochs 100 --no-cuda False --log-interval 100
"""

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    # reconstruction loss + KL divergence
    recon_loss = F.binary_cross_entropy(recon_x, x.reshape(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + KLD


def train(train_loader, args, device, model, optimizer, epoch):
    """Training for an epoch"""
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_fn(recon_batch, data, mu, logvar)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )
    
    train_loss /= len(train_loader.dataset)
    return train_loss 

def test(test_loader, args, device, model, epoch):
    """Test for an epoch"""
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_fn(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), f"./reconstruction_{epoch}.png", nrow=n)

    test_loss /= len(test_loader.dataset)
    return test_loss 


def train_epoch(train_loader, test_loader, args, device, model, optimizer, num_epoch=10, filename="checkpoint.pt"):
    """Main training function"""
    min_test_loss = float('inf')
    best_epoch = None 
    loss = [] 
    for epoch in range(1, num_epoch + 1):
        train_loss = train(train_loader, args, device, model, optimizer, epoch)
        test_loss = test(test_loader, args, device, model, epoch)

        print(f"====> Epoch: {epoch} Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
        loss.append([train_loss, test_loss])

        # Save checkpoint
        if test_loss < min_test_loss:
            min_test_loss = test_loss 
            best_epoch = epoch
            print("Saving checkpoint...")
            torch.save(model.state_dict(), filename)


    print(f"The minimum test loss of {min_test_loss:.3f} occurred after epoch {best_epoch}.")
    with open(f'./loss.npy', 'wb') as f:
        np.save(f, np.array(loss))


def main():
    # parsing arguments from command line 
    parser = argparse.ArgumentParser(
    description="VAE MNIST Example"
    )  # collect arguments passed to file
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    args = parser.parse_args()


    # set device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")  # Use NVIDIA CUDA GPU if available

    # initialize model
    model = VAE().to(device)

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    
    # load Fashion MNIST dataset
    kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
    torch.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=True, download=True,
                    transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, download=True,transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)


    # training 
    train_epoch(train_loader, test_loader, args, device, model, 
                optimizer, num_epoch=args.epochs, filename="checkpoint.pt")


if __name__ == "__main__":
    main()
