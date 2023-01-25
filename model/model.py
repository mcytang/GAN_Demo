import torch
from torch import nn
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, target_distribution, mu, std, Nsample = 32, depth = 4, width = 64):
        """
        input x = [Nbatch, Nseed, Nsample]
        where: 
            Nbatch is the batch size
            Nseed is the size of the random seed (ie channels)
            Nsample is number of numbers to be generated
        """
        super(Generator, self).__init__()
        self.mu = mu 
        self.std = std

        self.target_distribution = target_distribution
        self.Nsample = Nsample
        self.depth = depth
        self.activation = nn.ReLU()

        self.width = width

        setattr(self, 'linear0', nn.Linear(self.Nsample, self.width))
        for i in range(1,self.depth-1):
            setattr(self, 'linear{}'.format(i), nn.Linear(self.width,self.width))
        setattr(self, 'linear{}'.format(self.depth-1), nn.Linear(self.width, self.Nsample))
        # initialise weights
        for i in range(self.depth):
            layer = getattr(
                self,
                'linear{}'.format(i)
            )
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, x): 
        for i in range(self.depth):
            x = getattr(self, 'linear{}'.format(i))(x)
            if i < self.depth - 1: 
                x = self.activation(x)
        return x



class Discriminator(nn.Module): 
    def __init__(self, Nsample, inChannels = 16, depth = 4, niter = 1, loss = 'LSGAN'):
        """
        input x = [Nbatch, 1, Nsample]
        where: 
            Nbatch is the batch size
            Nseed is the size of the random seed
        """
        super(Discriminator, self).__init__()
        self.Nsample = Nsample
        self.inChannels = inChannels
        self.depth = depth
            
        self.activation = nn.ReLU()

        self.fc0 = nn.Linear(self.Nsample, self.inChannels)
        for i in range(1, self.depth):
            setattr(
                self,
                'fc{}'.format(i),
                nn.Linear(self.inChannels // 2 ** (i-1), self.inChannels // 2 ** i)
            )
        # self.fc0 = spectral_norm(nn.Linear(self.Nsample, self.inChannels), n_power_iterations=niter)
        # for i in range(1, self.depth):
        #     setattr(
        #         self,
        #         'fc{}'.format(i),
        #         spectral_norm(nn.Linear(self.inChannels // 2 ** (i-1), self.inChannels // 2 ** i),
        #          n_power_iterations=niter)
        #     )

    def forward(self, y): 
        for i in range(self.depth-1):
            y =  getattr(self, 'fc{}'.format(i))(y)
            if i<self.depth - 1:
                y = self.activation(y)
        return y 