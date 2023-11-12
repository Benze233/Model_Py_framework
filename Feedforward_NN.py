from os import path
import torch
import numpy as np
from torch.nn.functional import mse_loss
from torch.utils.data import dataloader,TensorDataset
from sklearn.model_selection import train_test_split
from typing import List
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader,Dataset
class Feedforward_NN(nn.Module):
    def __init__(self,num_state:int,num_control:int, layersize: List[int]):
        super(Feedforward_NN, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state = num_state
        self.control = num_control
        self.layer_sizes = layersize
        inputsize = self.state+self.control
        self.network = nn.Sequential()
        for i in range(len(layersize)):
            if i==0:
                self.network.append(nn.Linear(inputsize,layersize[i]))
            else:
                self.network.append(nn.Linear(layersize[i-1],layersize[i]))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(layersize[-1],self.state))
    def forward(self,input):
        output = self.network(input)
        return output
    def train_Model(self,train_in:np.array,train_out:np.array,MaxEpoch:int,batchsize:int,learning_rate,val_ratio = 0.2):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_in = torch.from_numpy(train_in).to(device, dtype=torch.float32)
        train_out = torch.from_numpy(train_out).to(device, dtype=torch.float32)
        trainx, valx, trainy, valy = train_test_split(train_in, train_out, test_size=val_ratio)
        train_data = TensorDataset(trainx, trainy)
        val_data = TensorDataset(valx,valy)
        train_loader = DataLoader(train_data, batch_size=batchsize)
        val_loader = DataLoader(val_data, batch_size=batchsize)

        # Specify lossfunction
        loss_func = nn.MSELoss().to(device)
        # Specify optimizer
        optimizer = Adam(self.parameters(), lr=learning_rate)


        loss_hist = []
        val_loss = []
        for epoch in range(MaxEpoch):
            tmp_loss = []
            self.train()
            for idx, (x_batch, y_batch) in enumerate(train_loader, 1):
                yhat = self.forward(x_batch)
                batch_loss = loss_func(yhat, y_batch)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                tmp_loss.append(batch_loss)
            loss_hist.append(sum(tmp_loss) / idx)
            tmp_loss = []
            with torch.no_grad():
                self.eval()
                for idx, (x_batch, y_batch) in enumerate(val_loader, 1):
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    yhat = self.forward(x_batch)
                    batch_loss = loss_func(yhat, y_batch)
                    tmp_loss.append(batch_loss)
                val_loss.append(sum(tmp_loss) / idx)
            if (epoch + 1) % int(0.1*MaxEpoch) == 0:
                print(f"Epoch = {epoch + 1},train loss = {loss_hist[epoch]}, val loss = {val_loss[epoch]}")
        if device == 'cuda':
            loss_hist = torch.tensor(loss_hist).cuda().data.cpu().numpy()
            val_loss = torch.tensor(val_loss).cuda().data.cpu().numpy()
        else:
            loss_hist = torch.tensor(loss_hist).detach().numpy()
            val_loss = torch.tensor(val_loss).detach().numpy()
        return loss_hist, val_loss

    def save(self, base_name="RNN_model"):
        """Saves model to file system for use later

        Args:
            base_name (str): the name of the model
        """
        file_name = path.join("./Models", base_name+".pt")
        dict_to_save = {}
        dict_to_save['model_params'] = self.state_dict()
        dict_to_save['n_actions'] = self.control
        dict_to_save['n_states'] = self.state
        dict_to_save['layer_sizes'] = self.layer_sizes

        torch.save(dict_to_save, file_name)

    def load(base_name="RNN_model"):
        """Loads model from file system

        Args:
            base_name (str): the name of the model

        Returns:
            Actor: the loaded actor model
        """
        file_name = path.join("./Models", base_name+".pt")

        loaded_dict = torch.load(file_name,map_location=torch.device('cpu'))
        RNN_model = Feedforward_NN(loaded_dict['n_states'], loaded_dict['n_actions'], loaded_dict['layer_sizes'])
        RNN_model.load_state_dict(loaded_dict['model_params'])

        return RNN_model






