# seed 
seed = 2

# Generator parameters
target_distribution = 'Gaussian' # choose in {'Gaussian', 'Exponential'}
mu = 0 # distribution mean / parameter
std = 1
Nsample = 64
inChannels = 64
depth = 4

# Discriminator
D_inChannels = 32
D_depth = 8

#Training
loss = 'LSGAN' # choose in {'GAN', 'LSGAN'}
G_lr = 1e-2
D_lr = 1.1e-2
Nepochs = int(3e4)
batchSize = 8
drop_rate = int(2e4)
drop_ratio = 0.1
fig_freq = 1e4
save_freq = Nepochs // 2

show_fig = True
