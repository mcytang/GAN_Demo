def main(name, checkpoint = None, Nbatch = 128, maxiter = int(1e4)):
    import torch 
    import torch.distributions as d 
    from matplotlib import pyplot as plt
    from utils.helpers import samples_to_dist

    path = 'TrainedModels/' + name

    if checkpoint is None: 
        tmp = ''
    else: 
        tmp = '_{}'.format(checkpoint)

    G = torch.load(path + '/G{}.pt'.format(tmp))
    D = torch.load(path + '/D{}.pt'.format(tmp))

    atts = ['inChannels', 'depth']
    print('G args:')
    for att in atts:
        print(att, getattr(G, att))
    print('D args:')
    for att in atts: 
        print(att, getattr(D, att))

    if G.target_distribution == 'Gaussian':
        distribution = d.normal.Normal( G.mu * torch.ones(Nbatch, 1, G.Nsample), G.std * torch.ones(Nbatch, 1, G.Nsample))
    elif G.target_distribution == 'Exponential':
        distribution = d.exponential.Exponential(G.mu * torch.ones(Nbatch, 1, G.Nsample))


    f, ax = plt.subplots()
    if G.target_distribution == 'Exponential':
        xlims = [0, 10]
    elif G.target_distribution == 'Gaussian':
        xlims = [-3.5,3.5]

    ax.set_xlim(xlims)
    
    def update_generator(Gx, z, ylim):
        ax.cla()
        ax.set_title('Testing GAN number generator')
        xs, ys = samples_to_dist(Gx, xlims, False)
        ax.plot(xs, ys, label = 'GAN generated dist.', color = 'orange')
        xs, ys = samples_to_dist(z, xlims, False)
        ax.plot(xs, ys, label = 'True dist.')
        ax.legend(loc = 'upper right')
        ylim = max(ylim, max(ys))
        ax.set_ylim([0, ylim])
        plt.pause(1e-1)
        return ylim

    ylim = 0
    for i in range(maxiter):
        z = distribution.rsample()
        Gx = G(torch.rand((Nbatch, G.Nsample))*2 - 1)

        ylim = update_generator(Gx, z, ylim)
    
if __name__ == '__main__':
    from sys import argv
    if len(argv) < 3:
        checkpoint = None
    else: 
        checkpoint = argv[2]
    main(argv[1], checkpoint)