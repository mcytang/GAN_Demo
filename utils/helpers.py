def myprint(X):
    msg = ''
    msg += '[{}/{}]'.format(X.epoch, X.Nepochs)
    msg += '[G_loss: {:.3f}]'.format(X.G_losses[-1])
    msg += '[D_loss: {:.3f}]'.format(X.D_losses[-1])
    print(msg, end = '\r')

def save_model(G, D, P):
    import os
    import torch 
    from datetime import datetime

    tmp = datetime.now().strftime('%d_%m_%Y-%H_%M')
    os.mkdir(r'TrainedModels/{}'.format(tmp))
    torch.save(G, r'TrainedModels/{}/G.pt'.format(tmp))
    torch.save(D, r'TrainedModels/{}/D.pt'.format(tmp))
    if P.save_fig:
        plt.savefig(r'TrainedModels/{}/fig.png'.format(tmp))
    with open(r'TrainedModels/{}/parameters.txt'.format(tmp)) as f:
        s=''
        s+='seed: {}\n'.format(P.seed)
        s+='\n'
        s+='target_distribution: {}\n'.format(P.target_distribution)
        s+='mu: {}\n'.format(P.mu)
        s+='std: {}\n'.format(P.std)
        s+='Nsample: {}\n'.format(P.Nsample)
        s+='Nseed: {}\n'.format(P.Nseed)
        s+='inChannels: {}\n'.format(P.inChannels)
        s+='depth: {}\n'.format(P.depth)
        s+='\n'
        s+='D_inChannels: {}\n'.format(P.D_inChannels)
        s+='D_depth: {}\n'.format(P.D_depth)
        s+='\n'
        s+='lr: {}\n'.format(P.lr)
        s+='Nepochs: {}\n'.format(P.Nepochs)
        s+='batchSize: {}\n'.format(P.batchSize)
        s+='drop_rate: {}\n'.format(P.drop_rate)
        s+='drop_ratio: {}\n'.format(P.drop_ratio)
        s+='fig_freq: {}\n'.format(P.fig_freq)
        s+='\n'
        s+='show_fig: {}\n'.format(P.show_fig)
import numpy as np
def samples_to_dist(samples, xlims):
    inc = 0.2
    xs = np.arange(xlims[0], xlims[1], inc)
    ys = []
    samples= samples.detach().cpu()
    for x in xs:
        idx = (samples >= x) * (samples < x + inc)
        ys.append(idx.sum().cpu().numpy()/ samples.numel())

    if max(ys) > 0:
        ys = ys / max(ys)
    return xs, ys


from matplotlib import pyplot as plt
class plot_helper():
    def __init__(self, target_distribution, ylim = 1):
        self.f, ax = plt.subplots(1,2)
        self.colours = ['tab:blue', 'tab:orange']
        self.ax1 = ax[1]
        self.ax1.set_ylabel('G_loss', color = self.colours[0])
        self.ax1.set_xlabel('iterations')
        self.ax1.set_ylim([0,ylim])
        self.ax1.tick_params(axis = 'y', labelcolor = self.colours[0])
        self.ax1.grid(axis='y')
        if target_distribution == 'Exponential':
            self.xlims = [0, 10]
        elif target_distribution == 'Gaussian':
            self.xlims = [-3.5,3.5]

        self.ax = ax[0]
        self.ax.set_xlim(self.xlims)
        self.ax.set_ylim([0, 1.1])
        self.legend_exists = False
        plt.pause(1e-3)

    def update_generator(self, Gx, z):
        self.ax.cla()
        xs, ys = samples_to_dist(Gx, self.xlims)
        self.ax.plot(xs, ys, color = 'orange')
        xs, ys = samples_to_dist(z, self.xlims)
        self.ax.bar(xs, ys)
        plt.pause(1e-3)

    def update_loss(self, X):
        self.ax1.plot(X.G_losses, label = 'G',color = self.colours[0], linewidth = 0.5)
        self.ax1.plot(X.D_losses, label = 'D',color = self.colours[1], linewidth = 0.5)
        self.ax1.plot(X.real_losses, label = 'D_real',color = 'green', linewidth = 0.5)
        self.ax1.plot(X.fake_losses, label = 'D_fake',color = 'red', linewidth = 0.5)
        if not self.legend_exists: 
            self.ax1.legend()
            self.legend_exists = True
        plt.pause(1e-3)

