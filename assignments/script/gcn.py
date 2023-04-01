"""
baseline Graph convolutional network from Kipf & Welling
https://arxiv.org/pdf/1609.02907.pdf
same as `from torch_geometric.nn import GCNConv`


# set torch version and cuda version
import torch; print(torch.__version__)
TORCH="1.10.0"
CUDA="cu111"

pip install torch-scatter -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html -q
pip install torch-sparse -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html -q
pip install torch-cluster -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html -q
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html -q
pip install torch-geometric -q
"""

import torch
from torch import nn 
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    """build a simple message passing layer from Kipf & Welling"""

    def __init__(self, in_channels, out_channels):
      # aggregation function can be mean, add, max, min
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 4).

        self.lin = torch.nn.Linear(in_channels, out_channels) # linear layer (Step 2)

    def forward(self, x, edge_index):
        """
        x: node embeddings, shape [N, in_channels]  
        edge_index: shape [2, E]
        N: number of nodes
        E: number of edges
        """

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform neiboring node feature matrix by a weight matrix W
        x = self.lin(x)

        # Step 3: Compute normalization. 
        # row shape [num_edge], col shape [num_edge]
        row, col = edge_index

        # Computes the (unweighted) degree matrix 
        deg = degree(col, x.size(0), dtype=x.dtype)

        # compute square root of inverse of degree matrix
        deg_inv_sqrt = deg.pow(-0.5)

        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Compute normalization coefficients.
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        # x: node embeddings, norm: normalization coefficients
        # propagate function internally calls step 3 message(), step 4 aggregate() and step 5 update()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)),
                              x=x, norm=norm)

    def message(self, x_j, norm):
        """
        normalize the neighboring node features x_j by number of neighbor nodes in neighborhood

        @param:
        x_j: a lifted tensor has shape [E, out_channels], 
          contains the source node features of each edge. i.e., the neighbors of each node
        norm: normalization coefficients, shape [E, ]
        """

        # Step 3: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
      """
      Updates node embeddings using function γ for each node with aggregated message

      @param
      aggr_out: output of step 4 aggregate(), has shape [N, out_channels]
      """

      # Step 5: Return new node embeddings.
      return aggr_out

 
class GINConv(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_size):
        super(GINConv, self).__init__(aggr='add')  # Initialize the superclass with 'add' aggregation
        # Define the MLP used for feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_channels)
        )  
        # Initialize the epsilon parameter
        self.eps = nn.Parameter(torch.Tensor([0]))  

    def forward(self, x, edge_index):
        # Propagate the messages through the graph
        out = self.propagate(edge_index, x=x)
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r 
        return self.mlp(out)

    def message(self, x_j):
        """No normalization"""
        return x_j 

    def update(self, aggr_out):
        # Update node features based on the aggregated messages
        return aggr_out



