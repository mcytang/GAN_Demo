from torch import nn

class Generator(nn.Module):
    def __init__(self, target_distribution, mu, std, Nsample = 32, depth = 4, inChannels = 64):
        """
        input x = [Nbatch, Nsample]
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

        self.inChannels = inChannels

        setattr(self, 'linear0', nn.Linear(self.Nsample, self.inChannels))
        setattr(self, 'norm0', nn.BatchNorm1d(self.inChannels))
        for i in range(1,self.depth-1):
            setattr(self, 'linear{}'.format(i), nn.Linear(self.inChannels,self.inChannels))
            setattr(self, 'norm{}'.format(i), nn.BatchNorm1d(self.inChannels))
        setattr(self, 'linear{}'.format(self.depth-1), nn.Linear(self.inChannels, self.Nsample))
        # initialise weights
        for i in range(self.depth):
            layer = getattr(
                self,
                'linear{}'.format(i)
            )
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, x): 
        for i in range(self.depth-1):
            x = getattr(self, 'linear{}'.format(i))(x)
            x = self.activation(x)
            x = getattr(self, 'norm{}'.format(i))(x)
        x = getattr(self, 'linear{}'.format(self.depth-1))(x)
        return x


class Discriminator(nn.Module): 
    def __init__(self, Nsample, inChannels = 16, depth = 2):
        """
        input x = [Nbatch, Nsample]
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

    def forward(self, y): 
        for i in range(self.depth-1):
            y =  getattr(self, 'fc{}'.format(i))(y)
            if i<self.depth - 1:
                y = self.activation(y)
        return y 