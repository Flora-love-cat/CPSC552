"""
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
MNIST dataset of handwritten digits
"""

import torch
from torch import nn, optim # neural network modules and optimizer
from torch.autograd import Variable # add gradients to tensors
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels


# ##################################
# defining the model 
# ##################################
class FCN(nn.Module):
    """
    method 1: define a python class, which inherits the rudimentary functionality of a neural network from nn.Module

    method 2: you can create an equivalent model to FCNN above using nn.Sequential()
    model = nn.Sequential(nn.Linear(num_input, n_hidden_1),
                           nn.Sigmoid(),
                           nn.Linear(n_hidden_1, n_hidden_2),
                           nn.Sigmoid(),
                           nn.Linear(n_hidden_2, num_classes))
    
    """
    def __init__(self, input_dim=784, output_dim=10, p=0, n_hidden_1=64, n_hidden_2=32):
        super(FCN, self).__init__()
        
        # As you'd guess, these variables are used to set the number of dimensions coming in and out of the network. We supply them when we initialize the neural network class.
        # Adding them to the class as variables isn't strictly necessary in this case -- but it's good practice to do this book-keeping, should you need to reference the input dim from somewhere else in the class.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p = p # dropout level

        # a single call of nn.Linear creates a single fully connected layer with the specified input and output dimensions. 
        # All of the parameters are created automatically.
        self.layer1 = nn.Linear(input_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, output_dim)

        # You can find many other nonlinearities on the PyTorch docs.
        self.nonlin1 = nn.Sigmoid() # nn.Softplus(), nn.ELU(), nn.Tanh()
        self.nonlin2 = nn.ReLU()

    def forward(self, x):
        """
        When you give your model data e.g., by running `model(data)`, 
        the data gets passed to this forward function -- the network's forward pass.
        You can very easily pass this data into your model's layers, reassign the output to the variable x, and continue.
        Play with position of nonlinearity, and number of layers
        """
        h1 = nn.Dropout(p=self.p)(self.nonlin1(self.layer1(x)))
        h2 = nn.Dropout(p=self.p)(self.nonlin2(self.layer2(h1)))
        o = nn.Dropout(p=self.p)(self.nonlin3(self.layer3(h2)))

        return o 


def training(device, model, trainloader, optimizer,loss_fn):
    """Train model for an epoch, return batch loss and accuracy"""
    # turn on training mode
    model.train()
    # iterate over batches
    for batch_indx, (data, labels) in enumerate(trainloader):
        # 1 attach data and label to device
        data = data.to(device)      # (128, 784)
        labels = labels.to(device)  # (128,)
        # 2 zero out gradient
        optimizer.zero_grad()
        # 3 feed data to model
        out = model(data)  # (128,10)
        # parameters for compute lp norm
        params = [param.data.clone().detach().requires_grad_(True)\
                    for param in model.parameters() if param.requires_grad]  
        regularization = 0.01 * lp_reg(params, p=2) 
        loss = loss_fn(out, labels)
        # 4 compute a loss by comparing output to actual labels 
        criterion = loss + regularization 
        # 5 backpropogate loss
        criterion.backward()
        # 6 update model parameters for a step.
        optimizer.step() 
    
    acc = get_accuracy(out, labels)

    return loss.item(), acc

def evaluate(device, model, testloader,loss_fn):
    """Evaluate model for an epoch"""
    model.eval()   # turn on evaluation mode: disable dropout and has batch norm use the entire population statistics
    with torch.no_grad: # disables tracking of gradients in autograd
        # iterate over batches
        for batch_indx, (data, labels) in enumerate(testloader):
            # 1 attach data and label to device
            data = data.to(device)      # (128, 784)
            labels = labels.to(device)  # (128,)
            # 2 feed data to model
            out = model(data)  # (128,10) 
            # 3 compute a loss by comparing output to actual labels 
            loss = loss_fn(out, labels)
    
    acc = get_accuracy(out, labels)
    return loss.item(), acc

def train(device, model,trainloader, testloader, 
          optimizer,loss_fn,num_epochs=10, filename=1):  
    """
    Main training function
    INPUT:
    model: an untrained pytorch model
    trainloader: The training data, in a dataloader for easy batch iteration.
    testloader: The testing data, in a dataloader for easy batch iteration.
    optimizer: the model optimizer, initialized with a learning rate.
    loss_fn: e.g. Cross Entropy loss or Mean Squared Error.
    num_epochs: number of epochs
    filename: name for saved model, default to "checkpoint.pt"
    
    return: 
    metrics: np.array([train_acc, test_acc], ..)
        training and test accuracy vs epoch
    losses: np.array([train_loss, test_loss])
        training and test loss vs epoch
    model: the trained model
    """ 
    metrics = [[0,0]] # training accuracy and test accuracy
    losses = [] # training loss and test loss
    best_test_acc = 0. 
    best_epoch = None 
    
    # iterate over epochs
    for epoch in range(num_epochs):
        train_loss, train_acc = training(device, model, trainloader, optimizer,loss_fn)
        test_loss, test_acc = training(device, model, testloader,loss_fn)
            
        metrics.append([train_acc, test_acc])
        losses.append([train_loss, test_loss])
        
        # print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"===>  Epoch {epoch}")
            print(f"      train accuracy: {train_acc:.3f}\t test accuracy: {test_acc:.3f}")
            print(f"      train loss: {train_loss:.3f}\t test loss: {test_loss:.3f}")
        
        if test_acc > best_test_acc:
            best_test_acc, best_epoch = test_acc, epoch
            print("saving checkpoint....")
            torch.save(model.state_dict(), f'./checkpoint{filename}.pt')
    
    print(f"The best validation accuracy of {best_test_acc:.3f} occurred after epoch {best_epoch}.")

    with open(f'./metrics{filename}.npy', 'wb') as f:
        np.save(f, np.array(metrics))
        np.save(f, np.array(losses))
    
    return np.array(metrics), np.array(losses), model


# ##################################
# helper functions
# ##################################
def get_accuracy(output, targets):
    """calculates accuracy from model output and targets
    """
    output = output.detach()
    predicted = output.argmax(-1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / output.size(0) * 100

    return accuracy


def to_one_hot(y, c_dims=10):
    """converts a N-dimensional vector to a NxC dimnensional one-hot encoding
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    c_dims = c_dims if c_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], c_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def lp_reg(params,p=1):
    """
    L-p norm regularization
    """
    sum = 0
    for w in params:
        if len(w.shape) > 1: # if this isn't a bias
            sum += torch.sum(w**p)
    return sum ** (1/p)


def plot_loss_acc(filename="1", title=None):
    """Plot train/test loss vs. epoch and train/test accuracy vs epoch"""
    with open(f'./metrics{filename}.npy', 'rb') as f:
        metrics = np.load(f,allow_pickle=True)
        losses = np.load(f,allow_pickle=True) 
    num_epochs = losses.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # plot loss
    axes[0].plot(range(num_epochs), losses[:,0], 'g', label="Train eror")
    axes[0].plot(range(num_epochs), losses[:,1], 'b', label="Test error")
    axes[0].set(xlabel="Epoch", ylabel='Error',
                title=f"FCN Error {title}") 
    # axes[0].set_ylim(0., 1.2)
    axes[0].legend(bbox_to_anchor=(1.02,1))
    
    # plot accuracy
    axes[1].plot(range(num_epochs+1), metrics[:,0], 'g', label="Train accuracy")
    axes[1].plot(range(num_epochs+1), metrics[:,1], 'b', label="Test accuracy")
    axes[1].set(xlabel="Epoch", ylabel='Error',
                title=f"FCN Accuracy {title}") 
    # axes[1].set_ylim(0, 100)
    axes[1].legend(bbox_to_anchor=(1.02,1))

    fig.savefig(f'./metics{filename}.png')
    plt.show()


def plt_conf_matrix(model, test_data, test_labels, filename):
    """Plot confusion matrix"""

    model.load_state_dict(torch.load(filename))
    test_preds = model(test_data).detach().argmax(dim=-1)

    cm = confusion_matrix(test_labels, test_preds)
    display_labels = unique_labels(test_labels, test_preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(
            cmap="OrRd", #"viridis"
            ax=None,
            xticks_rotation='horizontal',
            values_format=None,
            colorbar=True,
        )
    plt.title("Confusion matrix", y=1.1)
    plt.savefig(f'./confusion_matrix.png')
    plt.show()


def main():
    # ##################################
    # set hyperparameters
    # ##################################
    # Training Parameters
    batch_size = 128
    learning_rate = 0.001 
    num_epochs = 10 
    

    # Network Parameters
    dropout = 0.5
    num_input = 784  # MNIST data input (img shape: 28*28)
    num_classes = 10  # MNIST total classes (0-9 digits)
    n_hidden_1 = 64  # 1st layer number of neurons
    n_hidden_2 = 32  # 2st layer number of neurons
    
    # loss function
    CELoss = torch.nn.CrossEntropyLoss()

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # initialize model
    model = FCN(num_input, num_classes, p=dropout, n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2).to(device)
    
    # check model parameters, e.g., layer1.weight torch.Size([128, 784])
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)

    # initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    
    # ##################################
    # import data
    # ##################################

    # set seed for shuffle
    torch.manual_seed(42)
    # download the MNIST dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    # separate into data and labels
    # training data: 60k samples, 784 features
    train_data = mnist_trainset.data.to(dtype=torch.float32) # [60000, 28, 28]
    train_data = train_data.reshape(-1, 784) # [60000, 784]
    train_labels = mnist_trainset.targets.to(dtype=torch.long) # [60000]

    print("train data shape: {}".format(train_data.size()))
    print("train label shape: {}".format(train_labels.size()))

    # testing data: 2k samples, 784 features
    test_data = mnist_testset.data.to(dtype=torch.float32)[:2000] # [2000, 28, 28]
    test_data = test_data.reshape(-1, 784) # [2000, 784]
    test_labels = mnist_testset.targets.to(dtype=torch.long)[:2000] # [2000]

    print("test data shape: {}".format(test_data.size()))
    print("test label shape: {}".format(test_labels.size()))

    # create torch datasets
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    # creata torch dataloader
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    # ##################################
    # Training 
    # ##################################

    print(f"Training model >>>>>")
    train(device, model, 
          trainloader, testloader, 
          optimizer,
          CELoss,
          num_epochs=num_epochs, 
          filename="1")
    
    print(f"<<<<< Training model completed")

    # ##################################
    # Evaluation 
    # ##################################

    plot_loss_acc(filename="1") 
    plt_conf_matrix(model, test_data, test_labels, filename='checkpoint.pt')



if __name__ == '__main__':
    main()


