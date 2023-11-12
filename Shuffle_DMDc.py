import numpy as np
from mapminmax import mapminmax
from numpy.linalg import svd, inv
import os
from sklearn.utils import shuffle

class DMDc():
    def __init__(self,x1:np.array,x2:np.array,u:np.array,Shuffle = True):
        ### Accept data in row variable in column
        self.x1 = x1
        self.x2 = x2
        self.u = u
        self.omega = np.hstack((self.x1,self.u))
        if Shuffle:
            self.omega,self.x2 = shuffle(self.omega,self.x2)
        self.x1 = self.x1.T
        self.u = self.u.T
        self.omega = self.omega.T
        self.x2 = self.x2.T
        self.U_telda, self.S_telda, self.V_telda = svd(self.omega,full_matrices=False)
        self.V_telda = self.V_telda.T
    def Analyze(self):
        print(f"eigen value of S telda is {self.S_telda}")

    def identification(self,p):
        n = self.x1.shape[0]
        l = self.u.shape[0]
        U_telda = self.U_telda[:,0:p]
        S_telda = np.diag(self.S_telda[0:p])
        V_telda = self.V_telda[:,0:p]

        G = self.x2@V_telda@inv(S_telda)@U_telda.T
        A_hat = G[0:n,0:n]
        B_hat = G[0:n,n:]

        self.param = {}
        self.param["A"]=A_hat
        self.param["B"]=B_hat
        return self.param

    def save(self,filename = "DMDc_model"):
        path_direc = os.path.join("./Models/",filename+".npy")
        with open(path_direc,"wb") as f:
            np.savez(f,A=self.param["A"],B=self.param["B"])
