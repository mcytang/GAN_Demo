import torch 
from utils.helpers import myprint, plot_helper, save_model

class GANtrain():
    def __init__(
        self, 
        G_lr = 1e-3, 
        D_lr = 1e-3, 
        Nepochs = 64, 
        batchSize = 128,
        drop_rate = 16, 
        drop_ratio = 0.4,
        show_fig = False,
        fig_freq = 1000,
        save_freq = 1000
    ):
        self.G_lr = G_lr 
        self.D_lr = D_lr 
        self.Nepochs = Nepochs 
        self.batchSize = batchSize
        self.drop_rate = drop_rate
        self.drop_ratio = drop_ratio 
        self.show_fig = show_fig
        self.fig_freq = fig_freq
        self.save_freq = save_freq

        self.G_losses = []
        self.D_losses = []
        self.real_losses = []
        self.fake_losses = []

    def __call__(self, model_name, G, D, loss_func, mu, std):

        dim = (self.batchSize, G.Nsample)
        z_dim = (self.batchSize, G.Nsample)
        
        if G.target_distribution == 'Gaussian':
            distribution = torch.distributions.normal.Normal(mu * torch.ones(z_dim), std * torch.ones(z_dim))
        elif G.target_distribution == 'Exponential':
            distribution = torch.distributions.exponential.Exponential(mu * torch.ones(z_dim))

        G.train()
        D.train()
        
        G_optimiser = torch.optim.SGD(G.parameters(), lr = self.G_lr)#, momentum = 0.1)
        G_scheduler = torch.optim.lr_scheduler.StepLR(G_optimiser, step_size=self.drop_rate, gamma=self.drop_ratio)

        D_optimiser = torch.optim.SGD(D.parameters(), lr = self.D_lr)
        D_scheduler = torch.optim.lr_scheduler.StepLR(D_optimiser, step_size=self.drop_rate, gamma=self.drop_ratio)
        
        #labels
        
        if self.show_fig:
            r = max(loss_func.real, loss_func.fake)
            f = plot_helper(G.target_distribution, (r ** 2 ) * 2)
        
        for n in range(self.Nepochs):
            self.epoch = n

            D_optimiser.zero_grad()

            ###################
            ## Discriminator ##
            ###################

            # get seed 
            x = torch.rand(dim)*2 - 1

            # generate
            Gx = G(x)

            # discriminate
            z = distribution.rsample()
            DGx = D(Gx)
            Dz = D(z)

            # compute loss
            w0 = (torch.rand_like(DGx)-0.5) * 1e-2
            w1 = (torch.rand_like(Dz)-0.5) * 1e-2
            fake = torch.ones_like(DGx) * loss_func.fake + w0
            real = torch.ones_like(Dz) * loss_func.real + w1
            fake_loss = loss_func(DGx, fake) 
            real_loss = loss_func(Dz, real)
            D_loss = (real_loss + fake_loss ) / 2

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

            x = torch.rand(dim)
            DGx = D(G(x))# discriminate
            G_loss = loss_func(DGx, torch.ones_like(DGx) * loss_func.counterfeit)# compute loss

            G_loss.backward() #compute gradients
            G_optimiser.step() #update weights
            G_scheduler.step()

            self.G_losses.append(G_loss.clone().detach())

            if self.show_fig and n % self.fig_freq == 0:
                f.update_generator(Gx, z)
                f.update_loss(self)

            if n % self.save_freq == 0 and n > 0:
                save_model(model_name, G, D, self.show_fig, checkpoint=n)

            myprint(self)

        return G, D