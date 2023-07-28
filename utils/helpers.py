import torch 
import numpy as np
from matplotlib import pyplot as plt
import loss as l

def get_loss(name):
    if name == l.loss.GAN:
        return l.GAN_loss()
    elif name == l.loss.LSGAN: 
        return l.l2()

def myprint(X):
    msg = ''
    msg += '[{}/{}]'.format(X.epoch, X.Nepochs)
    msg += '[G_loss: {:.3f}]'.format(X.G_losses[-1])
    msg += '[D_loss: {:.3f}]'.format(X.D_losses[-1])
    print(msg, end = '\r')

def save_model(name, G, D, show_fig, checkpoint = None):

    if checkpoint is None:
        s = ''
    else: 
        s = '_{}'.format(checkpoint)

    torch.save(G, r'TrainedModels/{}/G{}.pt'.format(name,s))
    torch.save(D, r'TrainedModels/{}/D{}.pt'.format(name,s))
    if show_fig:
        plt.savefig(r'TrainedModels/{}/fig{}.png'.format(name,s))

def save_time(name, t):
    path = r'TrainedModels/{}/time.txt'.format(name)
    with open(path, 'w') as f: 
        s = 'Training time: {} seconds'.format(t)
        f.write(s)

def save_parameters(name, P):
    with open(r'TrainedModels/{}/parameters.txt'.format(name), 'w') as f:
        s=''
        s+='seed: {}\n'.format(P.seed)
        s+='\n'
        s+='target_distribution: {}\n'.format(P.target_distribution)
        s+='mu: {}\n'.format(P.mu)
        s+='std: {}\n'.format(P.std)
        s+='Nsample: {}\n'.format(P.Nsample)
        s+='inChannels: {}\n'.format(P.inChannels)
        s+='depth: {}\n'.format(P.depth)
        s+='\n'
        s+='D_inChannels: {}\n'.format(P.D_inChannels)
        s+='D_depth: {}\n'.format(P.D_depth)
        s+='\n'
        s+='loss: {}\n'.format(P.loss)
        s+='G_lr: {}\n'.format(P.G_lr)
        s+='D_lr: {}\n'.format(P.D_lr)
        s+='Nepochs: {}\n'.format(P.Nepochs)
        s+='batchSize: {}\n'.format(P.batchSize)
        s+='drop_rate: {}\n'.format(P.drop_rate)
        s+='drop_ratio: {}\n'.format(P.drop_ratio)
        s+='fig_freq: {}\n'.format(P.fig_freq)  
        s+='\n'
        s+='show_fig: {}\n'.format(P.show_fig)
        f.write(s)

def samples_to_dist(samples, xlims, normalize = True):
    inc = 0.05
    xs = np.arange(xlims[0], xlims[1], inc)
    ys = []
    samples= samples.detach().cpu()
    for x in xs:
        idx = (samples >= x- inc/2) * (samples < x + inc/2)
        ys.append(idx.sum().cpu().numpy()/ samples.numel())

    if normalize:
        if max(ys) > 0:
            ys = ys / max(ys)
    return xs, ys


class plot_helper():
    def __init__(self, target_distribution, ylim = 1):
        self.f, ax = plt.subplots(1,2)
        self.colours = ['tab:blue', 'tab:orange']
        self.ax1 = ax[1]
        self.ax1.set_ylim([0,ylim])
        if target_distribution == 'Exponential':
            self.xlims = [0, 10]
        elif target_distribution == 'Gaussian':
            self.xlims = [-5, 5]

        self.ax = ax[0]
        self.ax.set_xlim(self.xlims)
        self.ax.set_ylim([0, 1.1])
        plt.pause(1e-1)

    def update_generator(self, Gx, z):
        self.ax.cla()
        xs, ys = samples_to_dist(Gx, self.xlims)
        self.ax.plot(xs, ys, color = 'orange')
        xs, ys = samples_to_dist(z, self.xlims)
        self.ax.plot(xs, ys, color = 'blue')
        plt.pause(1e-1)

    def update_loss(self, X):
        
        self.ax1.cla() # clear to avoid accumulating data
        
        # format
        self.ax1.set_ylabel('G_loss', color = self.colours[0])
        self.ax1.set_xlabel('iterations')
        self.ax1.grid(axis='y')
        self.ax1.tick_params(axis = 'y', labelcolor = self.colours[0])

        # plot
        self.ax1.plot(X.real_losses, label = 'D_real',color = 'green', linewidth = 0.5)
        self.ax1.plot(X.fake_losses, label = 'D_fake',color = 'red', linewidth = 0.5)
        self.ax1.plot(X.D_losses, label = 'D',color = self.colours[1], linewidth = 0.5)
        self.ax1.plot(X.G_losses, label = 'G',color = self.colours[0], linewidth = 0.5)
        self.ax1.legend()
        plt.pause(1e-1)

