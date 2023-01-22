from torch import nn
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, target_distribution, mu, std, Nseed = 16, Nsample = 32, inChannels = 16, depth = 3):
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
        self.Nseed = Nseed
        self.Nsample = Nsample
        self.inChannels = inChannels
        self.depth = depth
        self.activation = nn.LeakyReLU()

        self.fc0 = nn.Linear(self.Nseed * self.Nsample, self.inChannels)
        self.bnorm0 = nn.BatchNorm1d(1)
        for i in range(1, self.depth-1):
            setattr(
                self,
                'fc{}'.format(i),
                nn.Linear(self.inChannels, self.inChannels)
            )
            setattr(
                self, 
                'bnorm{}'.format(i),
                nn.BatchNorm1d(1)
                )
        self.final = nn.Linear(self.inChannels, self.Nsample)

        # initialise weights
        for i in range(self.depth-1):
            layer = getattr(
                self,
                'fc{}'.format(i)
            )
            nn.init.kaiming_normal_(layer.weight)
    def forward(self, x): 
        x = x.reshape((x.shape[0], 1, self.Nseed * self.Nsample))
        for i in range(self.depth-1):
            x = getattr(self, 'fc{}'.format(i))(x)
            x = self.activation(x)
            x = getattr(self, 'bnorm{}'.format(i))(x)
        return self.final(x)

class Discriminator(nn.Module): 
    def __init__(self, Nsample, inChannels = 16, depth = 4, niter = 1):
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
        self.dropout = nn.Dropout(0.5)

        N = self.Nsample * self.inChannels
        self.fc0 = spectral_norm(nn.Linear(self.Nsample, N // 2), n_power_iterations=niter)
        for i in range(1, self.depth):
            setattr(
                self,
                'fc{}'.format(i),
                spectral_norm(nn.Linear(N // 2 ** i, N // 2 ** (i+1)),
                 n_power_iterations=niter)
            )

    def forward(self, y): 
        for i in range(self.depth):
            y = getattr(self, 'fc{}'.format(i))(y)
            if i == 0: y = self.dropout(y)
            if i<self.depth - 1:
                y = self.activation(y)
        return y