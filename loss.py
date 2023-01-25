import torch
import torch.nn as nn

class l2(nn.Module):
    def __init__(self):
        super(l2, self).__init__()
        self.r = 0.9
        self.real = self.r
        self.fake = -self.r
        self.counterfeit = self.r

        self.loss = nn.MSELoss()

    def forward(self, y, label):
        return self.loss(y, label) / (2*(self.r ** 2))

class GAN_loss(nn.Module):
    def __init__(self):
        super(GAN_loss,self).__init__()
        self.real = 0.1
        self.fake = 0.9
        self.counterfeit = 0.9

        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, y, label):

        return self.loss(y, label)