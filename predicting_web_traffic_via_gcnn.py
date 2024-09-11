
# importing libs
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

!pip install torch-geometric

from torch_geometric.datasets import WikipediaNetwork
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
import seaborn as sns
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from scipy.stats import norm

dataset = WikipediaNetwork(root='.', name='chameleon', transform=T.RandomNodeSplit(num_val=200, num_test=500))
data = dataset[0]

print(f'Dataset: {dataset}')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of unique features: {data.num_features}')
print(f'Number of classes: {dataset.num_classes}')

print(f'\nGraph:')
print(f'Edges are directed: {data.is_directed()}')
print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
print(f'Contains self-loops: {data.has_self_loops()}')

data.x.shape, data.y.shape

data.y  # these classes are saved as bins so we have to download and extract them to be able to convert them

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

url = 'https://snap.stanford.edu/data/wikipedia.zip'
with urlopen(url) as zip_url:
  with ZipFile(BytesIO(zip_url.read())) as zip_file:
    zip_file.extractall('.')

dataframe = pd.read_csv('/content/wikipedia/chameleon/musae_chameleon_target.csv')

dataframe.head(4)

# we convert the values to log of values, since the goal is to predict the log of average monthly traffic
values = np.log10(dataframe['target'])
data.y = torch.tensor(values)
data.y

dataframe['target'] = values
sns.distplot(dataframe['target'], fit=norm)

class GCN(torch.nn.Module):
  def __init__(self, dim_in, dim_h, dim_out):
    super().__init__()
    self.gcn1 = GCNConv(dim_in, dim_h*4)
    self.gcn2 = GCNConv(dim_h*4, dim_h*2)
    self.gcn3 = GCNConv(dim_h*2, dim_h)
    self.linear = torch.nn.Linear(dim_h, dim_out)

  def forward(self, x, edge_index):
    h = self.gcn1(x, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.gcn2(h, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.gcn3(h, edge_index)
    h = torch.relu(h)
    h = self.linear(h)
    return h

  def fit(self, data, epochs):
    optimizer = torch.optim.Adam(self.parameters(),
                                 lr=0.02,
                                 weight_decay=5e-4)
    self.train()
    for epoch in range(epochs+1):
      optimizer.zero_grad()
      out = self(data.x, data.edge_index)
      loss = F.mse_loss(out.squeeze()[data.train_mask], data.y[data.train_mask].float())  # mean squared of loss
      loss.backward()
      optimizer.step()
      if epoch % 20 == 0:
        val_loss = F.mse_loss(out.squeeze()[data.train_mask], data.y[data.train_mask])
        print(f"Epoch {epoch:>3} | Train loss: {loss:.5f} | Val loss: {val_loss:.5f}")

  def test(self, data):
    self.eval()
    out = self(data.x, data.edge_index)
    return F.mse_loss(out.squeeze()[data.train_mask], data.y[data.train_mask].float())

# create the model
gcn = GCN(dataset.num_features, 128, 1)
print(gcn)

# train
gcn.fit(data, epochs=200)

# evaluate
loss = gcn.test(data)
print(f'\nGCN test loss: {loss:.5f}\n')

from sklearn.metrics import mean_absolute_error, mean_squared_error
out = gcn(data.x, data.edge_index)
y_pred = out.squeeze()[data.test_mask].detach().numpy()  # squeezing the output to be in the same dimension as dataset
mse = mean_squared_error(data.y[data.test_mask], y_pred)
mae = mean_absolute_error(data.y[data.test_mask], y_pred)

print('=' * 43)
print(f'MSE = {mse:.4f} | RMSE = {np.sqrt(mse):.4f} | MAE = {mae:..4f}')


# getting the prediction for a single node 
single_node_index = data.test_mask.nonzero(as_tuple=True)[0][0].item()  # get the first node index from the test mask
single_node_prediction = out[single_node_index].item()  # extract the prediction for this node

# get the true value for this node for comparison
single_node_true_value = data.y[single_node_index].item()

print(f'Single Node Prediction: {single_node_prediction:.4f}')
print(f'True Value: {single_node_true_value:.4f}')










