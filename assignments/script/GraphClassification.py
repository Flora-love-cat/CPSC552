"""
Graph Classification with Graph Neural Networks

adapted from the PyTorch geometric tutorial 
https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html
"""
import numpy as np 
import torch
from torch import nn 
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Constant
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool


class GraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GraphClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.bn = nn.BatchNorm1d(hidden_channels, affine=False)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0, training=self.training)
        x = self.lin(x)
        return x


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    correct = 0
    for data in train_loader:  
        data = data.to(device)
        optimizer.zero_grad() 
        out = model(data.x, data.edge_index, data.batch)  
        loss = criterion(out, data.y)  
        loss.backward()  
        optimizer.step() 

        pred = out.argmax(dim=1)  
        correct += int((pred == data.y).sum()) 
    acc = correct / len(train_loader.dataset) 

    return loss.item(), acc

     
def test(model, test_loader, criterion, device):
    model.eval()

    correct = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y) 
        pred = out.argmax(dim=1)  
        correct += int((pred == data.y).sum()) 
    acc = correct / len(test_loader.dataset)  
    
    return loss.item(), acc

def train_epoch(model, train_loader, test_loader, criterion, optimizer, device, num_epoch=200):
    losses = []
    accs = []
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, num_epoch+1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        losses.append([train_loss, test_loss])
        accs.append([train_acc, test_acc])

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Train loss: {train_loss:.3f}, Test loss: {test_loss:.3f}')
            print(f'Train Accuracy: {train_acc:.3f}, Test Accuracy: {test_acc:.3f}')

        if test_acc > best_acc:
            best_acc = test_acc 
            best_epoch = epoch 
            torch.save(model.state_dict(), './checkpoint.pt')

    print(f"Best test accuracy {best_acc:.3f} occurred at epoch {best_epoch}")

    with open('./metrics.npy', 'wb') as f:
        np.save(f, losses)
        np.save(f, accs)
        

def main():
    #================================
    # Set hyperparameters
    #================================
    # training
    num_epoch = 200
    batch_size = 64
    lr = 0.01

    # model
    num_node_features = 1
    num_classes = 2
    hidden_channels = 64
    

    #================================
    # load data
    #================================
    dataset = TUDataset(
        root='data/TUDataset',
        name='REDDIT-BINARY',
        pre_transform=Constant() # the Reddit dataset has no node features of its own. This "Constant" pre-transform gives each node the value '1'.
        # If all goes according to plan, the GCN should be able to derive good graph representations from the connectivity of the graphs alone.
    )
    # print()
    # print(f'Dataset: {dataset}:')
    # print('====================')
    # print(f'Number of graphs: {len(dataset)}')
    # print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')
    
    # print()
    # data = dataset[0]  # Get the first graph object.
    # print(f"first graph object: {data}")
    # print('=============================================================')
    # # Gather some statistics about the first graph.
    # print(f'Number of nodes: {data.num_nodes}')
    # print(f'Number of edges: {data.num_edges}')
    # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    # print(f'Contains self-loops: {data.contains_self_loops()}')
    # print(f'Is undirected: {data.is_undirected()}')

    torch.manual_seed(12345) # for reproducibility
    dataset = dataset.shuffle()

    train_dataset = dataset[:1500]
    test_dataset = dataset[500:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #==============================================
    # initialize model, optimizer, loss function 
    #==============================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphClassifier(num_node_features=num_node_features, num_classes=num_classes, hidden_channels=hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    #==============================================
    # initialize model, optimizer, loss function 
    #=============================================
    train_epoch(model, train_loader, test_loader, criterion, optimizer, device, num_epoch=num_epoch)


if __name__ == "__main__":
    main()