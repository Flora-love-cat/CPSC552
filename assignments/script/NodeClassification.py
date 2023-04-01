"""
Node Classification with Graph Neural Networks

adapted from the PyTorch geometric tutorial 
https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html
"""
import numpy as np 
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, GINConv


class NodeClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_features):
        super(NodeClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train(model, data, criterion, optimizer, device):
    model.train()
    data = data.to(device)
    optimizer.zero_grad() 
    out = model(data.x, data.edge_index) 
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) 
    loss.backward()  
    optimizer.step() 
    
    pred = out.argmax(dim=1)  
    correct = pred[data.train_mask] == data.y[data.train_mask]  
    acc = int(correct.sum()) / int(data.train_mask.sum())  
    
    return loss.item(), acc 


def test(model, data, criterion, device):
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask]) 
    pred = out.argmax(dim=1)  
    correct = pred[data.test_mask] == data.y[data.test_mask]  
    acc = int(correct.sum()) / int(data.test_mask.sum())  
    
    return loss.item(), acc

def train_epoch(model, data, criterion, optimizer, device, num_epoch=200):
    losses = []
    accs = []
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, num_epoch+1):
        train_loss, train_acc = train(model, data, criterion, optimizer, device)
        test_loss, test_acc = test(model, data, criterion, device)
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
    lr = 0.01

    # model
    num_node_features = 1433
    num_classes = 7
    hidden_features = 16
    
    #================================
    # load data
    #================================

    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]  # Get the first graph object.
    # print()
    # print(f'Dataset: {dataset}:')
    # print('====================')
    # print(f'Number of graphs: {len(dataset)}')
    # print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')
    
    # print()
    # print(data)
    # print('=============================================================')
    # # Gather some statistics about the first graph.
    # print(f'Number of nodes: {data.num_nodes}')
    # print(f'Number of edges: {data.num_edges}')
    # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    # print(f'Contains self-loops: {data.contains_self_loops()}')
    # print(f'Is undirected: {data.is_undirected()}')

    #==============================================
    # initialize model, optimizer, loss function 
    #==============================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NodeClassifier(num_node_features=num_node_features, num_classes=num_classes, hidden_features=hidden_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    #==============================================
    # initialize model, optimizer, loss function 
    #=============================================
    train_epoch(model, data, criterion, optimizer, device, num_epoch=num_epoch)


if __name__ == '__main__':
    main()

