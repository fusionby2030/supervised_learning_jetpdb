from .base import ModelInterface
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from data import TorchDataset
from torch.utils.data import DataLoader
import functools 
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))

class FFNN(nn.Module): 
    def __init__(self, in_dim, hidden_dims): 
        super(FFNN, self).__init__()
        self.block = nn.ModuleList()
        for h_dim in hidden_dims:
            self.block.append(nn.Linear(in_dim, h_dim))
            self.block.append(nn.ReLU())
            in_dim = h_dim
        self.out = nn.Linear(in_dim, 1)
    def forward(self, x):
        for lay in self.block:
            x = lay(x)
        return self.out(x)


class FFNNInterface(ModelInterface):
    def __init__(self, params):
        self.params = params  
        # Build network 
        
        self.batch_size = params['batch_size']
        self.num_epochs = params['epochs']
        criterion_name = params['criterion']
        
        
        self.criterion = rgetattr(torch, criterion_name)()

    def initialize_model_and_optim(self, params): 
        self.model = None 
        in_d = params['mp_dim']
        hidden_dims = params['hidden_dims']
        optim_name = params['optimizer']
        lr = params['learning_rate']
        model = FFNN(in_d, hidden_dims)
        self.model = model.double()
        self.optimizer = rgetattr(torch, optim_name)(self.model.parameters(), lr)

        

    def train(self, data): 
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
        train_ds, val_ds, test_ds = TorchDataset(X_train, y_train), TorchDataset(X_val, y_val), TorchDataset(X_test, y_test)
        train_dl, val_dl, test_dl = DataLoader(train_ds, batch_size=self.batch_size), DataLoader(val_ds, batch_size=self.batch_size), DataLoader(test_ds, batch_size=self.batch_size)
        self.initialize_model_and_optim(self.params)
        optimizer = self.optimizer 
        model = self.model 
        min_val_loss = float("inf")
        for epoch in range(self.num_epochs):
            # Train iteration
            for batch_idx, batch in enumerate(train_dl): 
                in_mps, in_neped = batch
                optimizer.zero_grad()
                out_neped = model.forward(in_mps).squeeze()
                loss = self.criterion(in_neped, out_neped)

                loss.backward()
                optimizer.step()
            # Val iteration 
            for batch_idx, batch in enumerate(val_dl): 
                with torch.no_grad(): 
                    in_mps, in_neped = batch
                    out_neped = model.forward(in_mps).squeeze()
                    loss = self.criterion(in_neped, out_neped)
                if loss <= min_val_loss: 
                    # print(f'New best model {loss.item():.4f}')
                    self.model = model 

    def test(self, data): 
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
        # train_ds, val_ds, test_ds = TorchDataset(X_train, y_train), TorchDataset(X_val, y_val), TorchDataset(X_test, y_test)
        # train_dl, val_dl, test_dl = DataLoader(train_ds, batch_size=self.batch_size), DataLoader(val_ds, batch_size=self.batch_size), DataLoader(test_ds, batch_size=self.batch_size)

        with torch.no_grad(): 
            test_mps = torch.from_numpy(X_test)
            test_pred_neped = self.model.forward(test_mps).squeeze().numpy()
        # test_r2,test_mse, test_mape = calculate_metrics(y_test, test_pred_neped)
        return y_test, test_pred_neped


        


    

    