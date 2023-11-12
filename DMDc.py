import numpy as np
from mapminmax import mapminmax
from numpy.linalg import svd, inv
import os


class DMDc():
    def __init__(self,x:np.array,u:np.array):
        ### Need to make sure time series in column, variable in row
        self.x = x
        self.u = u
        self.X1 = self.x[:,0:-2]
        self.X2 = self.x[:,1:-1]
        self.U1 = self.u[:,0:-2]
        self.U_telda, self.S_telda, self.V_telda = svd(np.vstack((self.X1, self.U1)),full_matrices=False)
        self.V_telda = self.V_telda.T
        self.U_hat,self.S_hat,self.V_hat = svd(np.vstack((self.X2)),full_matrices = False)
        self.V_hat = self.V_hat.T
    def Analyze(self):
        print(f"eigen value of S telda is {self.S_telda}")

    def identification(self,p,r):
        n = self.X1.shape[0]
        l = self.U1.shape[0]
        U_telda = self.U_telda[:,0:p]
        S_telda = np.diag(self.S_telda[0:p])
        V_telda = self.V_telda[:,0:p]
        U_hat = self.U_hat[:,0:r]
        S_hat =  np.diag(self.S_hat[0:r])
        V_hat = self.V_hat[:,0:r]
        G = self.X2@V_telda@inv(S_telda)@U_telda.T
        A_hat = G[0:n,0:n]
        B_hat = G[0:n,n:]
        A_telda = U_hat.T@A_hat@U_hat
        B_telda = U_hat.T@B_hat
        self.param = {}
        self.param["A_hat"]=A_hat
        self.param["B_hat"]=B_hat
        self.param["A_telda"]=A_telda
        self.param["B_telda"]=B_telda
        return self.param

    def save(self,filename = "DMDc_model"):
        path_direc = os.path.join("./Models/", filename + ".npy")
        with open(path_direc,"wb") as f:
            np.savez(f,A_telda=self.param["A_telda"],B_telda=self.param["B_telda"],A_hat = self.param["A_hat"],B_hat = self.param["B_hat"])


