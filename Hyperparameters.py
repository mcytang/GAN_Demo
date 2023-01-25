# seed 
seed = 2

# Generator parameters
target_distribution = 'Gaussian' # choose in {'Gaussian', 'Exponential'}
mu = 0 # distribution mean / parameter
std = 1
Nsample = 32
Nseed = None
inChannels = 32
depth = 2

# Discriminator
D_inChannels = 64
D_depth = 3

#Training
loss = 'LSGAN' # choose in {'GAN', 'LSGAN'}
G_lr = 1e-3
D_lr = 2e-3
Nepochs = int(1e5)
batchSize = 512
drop_rate = Nepochs
drop_ratio = 0.2
fig_freq = Nepochs // 40
save_freq = Nepochs // 10

show_fig = True
