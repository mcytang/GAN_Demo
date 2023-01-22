import torch 
from torch import distributions as d
from utils.helpers import myprint, plot_helper
class GANtrain():
    def __init__(
        self, 
        lr = 1e-3, 
        Nepochs = 64, 
        batchSize = 128,
        drop_rate = 16, 
        drop_ratio = 0.4,
        show_fig = False,
        fig_freq = 1000
    ):
        self.lr = lr 
        self.Nepochs = Nepochs 
        self.batchSize = batchSize
        self.drop_rate = drop_rate
        self.drop_ratio = drop_ratio 
        self.show_fig = show_fig
        self.fig_freq = fig_freq

        self.G_losses = []
        self.D_losses = []
        self.real_losses = []
        self.fake_losses = []

    def __call__(self, G, D):

        torch.manual_seed(314)

        loss_func = lambda y, label: (torch.abs(y-label) ** 2).mean()

        G_dim = (self.batchSize, G.Nseed, G.Nsample)
        D_dim = (self.batchSize , G.Nseed, G.Nsample)
        z_dim = (self.batchSize, 1, G.Nsample)

        mu = G.mu 
        std = G.std

        if G.target_distribution == 'Gaussian':
            distribution = d.normal.Normal(mu * torch.ones(z_dim), std * torch.ones(z_dim))
        elif G.target_distribution == 'Exponential':
            distribution = d.exponential.Exponential(mu * torch.ones(z_dim))

        G.train()
        D.train()

        G_optimiser = torch.optim.Adam(G.parameters(), lr = self.lr)
        G_scheduler = torch.optim.lr_scheduler.StepLR(G_optimiser, step_size=self.drop_rate, gamma=self.drop_ratio)

        D_optimiser = torch.optim.Adam(D.parameters(), lr = self.lr)
        D_scheduler = torch.optim.lr_scheduler.StepLR(D_optimiser, step_size=self.drop_rate, gamma=self.drop_ratio)
        
        #labels
        r = 1
        fake = -r
        real = r
        counterfeit = 0

        if self.show_fig:
            f = plot_helper(G.target_distribution, (r ** 2 ) * 2)

        for n in range(self.Nepochs):
            self.epoch = n

            D_optimiser.zero_grad()

            ###################
            ## Discriminator ##
            ###################

            # get seed 
            x = torch.rand(D_dim)

            # generate
            Gx = G(x)

            # discriminate
            z = distribution.rsample()
            DGx = D(Gx)
            Dz = D(z)

            # compute loss
            fake = torch.ones_like(DGx) * fake
            real = torch.ones_like(Dz) * real
            fake_loss = loss_func(DGx, fake) 
            real_loss = loss_func(Dz, real)
            D_loss = ( real_loss + fake_loss ) / 2

            # to optimiser step
            D_loss.backward() #compute gradients
            D_optimiser.step() #update weights
            D_scheduler.step()

            self.D_losses.append(D_loss.clone().detach())
            self.real_losses.append(real_loss.clone().detach())
            self.fake_losses.append(fake_loss.clone().detach())
                
            ###############
            ## Generator ##
            ###############

            G_optimiser.zero_grad()

            x = torch.rand(G_dim)
            DGx = D(G(x))# discriminate
            G_loss = loss_func(DGx, torch.ones_like(DGx) * counterfeit)# compute loss

            G_loss.backward() #compute gradients
            G_optimiser.step() #update weights
            G_scheduler.step()

            self.G_losses.append(G_loss.clone().detach())

            if self.show_fig and n % self.fig_freq == 0:
                f.update_generator(Gx, z)
                f.update_loss(self)

            myprint(self)


        return G, D