import torch
from torch import nn, flatten
from torch.nn import Linear,ReLU
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from os import path

class Koopman_AutoEncoder(nn.Module):
    def __init__(self,stateshape,hiddenshape):
        super(Koopman_AutoEncoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stateshape = stateshape
        self.hiddenshape = hiddenshape
        self.encoder = nn.Sequential(
            Linear(stateshape,8),
            ReLU(),
            Linear(8,16),
            ReLU(),
            Linear(16,hiddenshape)
        )
        self.decoder = Linear(hiddenshape,stateshape)
        self.decoder.bias.requires_grad = False
        with torch.no_grad():
            self.decoder.bias.fill_(0)

    def forward(self, input):
        coded = self.encoder(input)
        out = self.decoder(coded)
        return out

    def decoding(self, input: np.array):
        with torch.no_grad():
            input = torch.from_numpy(input).to(dtype=torch.float32)
            gen = self.decoder(input)
            return gen

    def give_C(self):
        C_matrix = self.decoder.weight
        C_matrix = torch.tensor(C_matrix).cuda().data.cpu().numpy()
        return C_matrix

    def encoding(self, input: np.array):
        with torch.no_grad():
            input = torch.from_numpy(input).to(dtype=torch.float32)
            code = self.encoder(input)
            return code

    def train_Model(self, train_in: np.array, train_out: np.array, MaxEpoch: int, batchsize: int, learning_rate,
                    val_ratio=0.2):

        train_in = torch.from_numpy(train_in.T).to(self.device, dtype=torch.float32)
        train_out = torch.from_numpy(train_out.T).to(self.device, dtype=torch.float32)
        trainx, valx, trainy, valy = train_test_split(train_in, train_out, shuffle=True, test_size=val_ratio)
        train_data = TensorDataset(trainx, trainy)
        val_data = TensorDataset(valx, valy)
        train_loader = DataLoader(train_data, batch_size=batchsize)
        val_loader = DataLoader(val_data, batch_size=batchsize)

        # Specify loss function
        loss_func = nn.MSELoss().to(self.device)
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
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    yhat = self.forward(x_batch)
                    batch_loss = loss_func(yhat, y_batch)
                    tmp_loss.append(batch_loss)
                val_loss.append(sum(tmp_loss) / idx)
            if (epoch + 1) % int(0.1 * MaxEpoch) == 0:
                print(f"Epoch = {epoch + 1},train loss = {loss_hist[epoch]}, val loss = {val_loss[epoch]}")

        if self.device == 'cuda':
            loss_hist = torch.tensor(loss_hist).cuda().data.cpu().numpy()
            val_loss = torch.tensor(val_loss).cuda().data.cpu().numpy()
        else:
            loss_hist = torch.tensor(loss_hist).detach().numpy()
            val_loss = torch.tensor(val_loss).detach().numpy()
        return loss_hist, val_loss

    def save(self, base_name="Autoencoder_Koopman_model"):
        """Saves model to file system for use later

        Args:
            base_name (str): the name of the model
        """
        file_name = path.join("./Models", base_name+".pt")
        dict_to_save = {}
        dict_to_save['model_params'] = self.state_dict()
        dict_to_save['stateshape'] = self.stateshape
        dict_to_save['hiddenshape'] = self.hiddenshape

        torch.save(dict_to_save, file_name)

    def load(base_name="Autoencoder_Koopman_model"):
        """Loads model from file system

        Args:
            base_name (str): the name of the model

        Returns:
            Actor: the loaded actor model
        """
        file_name = path.join("./Models", base_name+".pt")

        loaded_dict = torch.load(file_name,map_location=torch.device('cpu'))
        AE_model = Koopman_AutoEncoder(loaded_dict['stateshape'], loaded_dict['hiddenshape'])
        AE_model.load_state_dict(loaded_dict['model_params'])

        return AE_model
