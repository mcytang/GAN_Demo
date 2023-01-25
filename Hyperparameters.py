# seed 
seed = 2

# Generator parameters
target_distribution = 'Gaussian' # choose in {'Gaussian', 'Exponential'}
mu = 0 # distribution mean / parameter
std = 1
Nsample = 4
Nseed = None
width = 64
depth = 4

# Discriminator

D_inChannels = 64
D_depth = 4

#Training
loss = 'GAN' # choose in {'GAN', 'LSGAN'}
G_lr = 2e-3
D_lr = 1e-3
Nepochs = 50000
batchSize = 128
drop_rate = 50000
drop_ratio = 0.5
fig_freq = Nepochs // 40
save_freq = Nepochs // 10

show_fig = True
