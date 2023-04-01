"""
Autoencoder
"""
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import optim, nn 
from torchvision import transforms, datasets


class Autoencoder(nn.Module):
    """Both encoder and decoder have 4 layers with a tanh activation between each linear layer. 
        The shape of each layer is 784-1000-500-250-2-250-500-1000-784. 
        Use sigmoid activation function on output of the last hidden layer.
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.en_lin1 = nn.Linear(784, 1000)
        self.en_lin2 = nn.Linear(1000, 500)
        self.en_lin3 = nn.Linear(500, 250)
        self.en_lin4 = nn.Linear(250, 2)

        self.de_lin1 = nn.Linear(2, 250)
        self.de_lin2 = nn.Linear(250, 500)
        self.de_lin3 = nn.Linear(500, 1000)
        self.de_lin4 = nn.Linear(1000, 784)


    def encode(self, x):
        x = self.en_lin1(x)
        x = torch.tanh(x) 
        x = self.en_lin2(x)
        x = torch.tanh(x)
        x = self.en_lin3(x)
        x = torch.tanh(x)
        x = self.en_lin4(x) 

        return x

    def decode(self, z):
        z = self.de_lin1(z) 
        z = torch.tanh(z)  
        z = self.de_lin2(z)
        z = torch.tanh(z)
        z = self.de_lin3(z)
        z = torch.tanh(z)
        z = self.de_lin4(z)  
        z = torch.sigmoid(z) 

        return z  

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def train(device, model, loss_fn, optimizer, scheduler, train_loader):
    """
    Train model for an epoch, return batch training error

    param:
    model: an untrained pytorch model
    loss_fn: e.g. Cross Entropy loss or Mean Squared Error.
    optimizer: the model optimizer, initialized with a learning rate.
    scheduler: scheduler for Optimizer
    train_loader: training data, in a dataloader for easy iteration.
    """
    model.train()   # training mode
    
    for data, _ in train_loader:
        # reshape 28*28 images to 784 and attach data to device
        data = data.reshape(-1, 784).to(device)
        # reset gradient
        optimizer.zero_grad()
        # get predictions from model
        out = model(data).to(device)
        # Calculate the loss
        loss = loss_fn(out, data)
        # Find the gradients of our loss via backpropogation
        loss.backward()
        # Adjust accordingly with the optimizer
        optimizer.step()
        
    scheduler.step()    # update scheduler per epoch

    return loss.item()


def evaluate(device, model, loss_fn, test_loader):
    """
    Evaluates the given model on the given dataset.
    Returns average batch loss 
    
    model: a trained model
    """
    model.eval() # set model to evaluation mode

    # disables backpropogation, which makes the model run much more quickly.
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.reshape(-1, 784).to(device)
            out = model(data) 
            loss = loss_fn(out, data)

    return loss.item()


def train_epoch(train_params, device, model, loss_fn, optimizer, scheduler, train_loader, test_loader):
    losses = [] # store training loss and test loss
    min_test_loss = np.inf # initialize min test loss to be infinity
    best_epoch = None # record which epoch reaches the min test loss
    
    for epoch in range(train_params['num_epochs']):

        train_loss = train(device, model, loss_fn, optimizer, scheduler, train_loader)
        test_loss = evaluate(device, model, loss_fn, test_loader)
        losses.append([train_loss, test_loss])
        
        # Give status reports every 5 epochs
        if epoch % 5 == 0:
            print(f"===> EPOCH {epoch}. Progress: {epoch/train_params['num_epochs']*100}%. ")
            print(f" Train loss: {train_loss:.3f}. Test loss: {test_loss:.3f}")

        # Save checkpoint
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            best_epoch = epoch

            print("Saving checkpoint...")
            torch.save(model.state_dict(), f'./checkpoint.pt')

    print(f"Min test loss {min_test_loss:.3f} occurred after epoch {best_epoch}.")

def plot_loss(filename=None, title=None):
    """
    plot training loss and test loss 
    """
    with open(f'./loss.npy', 'rb') as f:
        loss = np.load(f, allow_pickle=True)
    plt.figure(figsize=(8, 6))
    plt.plot(loss[:, 0], 'g', label="Train loss")
    plt.plot(loss[:, 1], 'b', label="Test loss")
    plt.title(title, size=20, y=1.1)
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.show()


def main():

    train_params = {'batch_size': 128, 
                    'num_epochs': 30, 
                    'lr': 0.001,
                    'weight_decay': 0.01,
                    'step_size': 5, 
                    'gamma': 0.1
                    }
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
    # initialize Autoencoder model
    model = Autoencoder().to(device) 
    
    # initialize optimizer, and set learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr = train_params['lr'], weight_decay=train_params['weight_decay'])
    # initialize schedular
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_params['step_size'], gamma=train_params['gamma'])
    
    # initialize Mean Squared Error loss function
    loss_fn = torch.nn.MSELoss()

    # load training set and test set 
    mnist_train = datasets.MNIST(root='data',train=True,download=True,transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root='data',train=False,download=True,transform=transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=train_params['batch_size'])
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=train_params['batch_size'])

    # train
    train_epoch(train_params, device, model, loss_fn, optimizer, scheduler, train_loader, test_loader)
    
    # evaluation
    plot_loss()



if __name__ == '__main__':
    main()
