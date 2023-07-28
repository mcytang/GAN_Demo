from loss import loss as l

# seed 
seed = 4

# Generator parameters
target_distribution = 'Gaussian' # choose in {'Gaussian', 'Exponential'}
mu = 2 # distribution mean / parameter
std = 1
Nsample = 32
inChannels = 32
depth = 2

# Discriminator
D_inChannels = 16
D_depth = 4

#Training
loss = l.LSGAN # choose in {'GAN', 'LSGAN'}
G_lr = 1e-2
D_lr = 1e-2
Nepochs = int(3e4)
batchSize = 4
drop_rate = int(2e4)
drop_ratio = 0.2
fig_freq = 1e4
save_freq = Nepochs // 2

show_fig = True
