# seed 
seed = 123

# Generator parameters
target_distribution = 'Gaussian'
mu = 0
std = 1
Nsample = 2 ** 3
Nseed = 1
inChannels = 16
depth = 8

# Discriminator

D_inChannels = 256
D_depth = 11

#Training
lr = 1e-3
Nepochs = 50000
batchSize = 32
drop_rate = 50000
drop_ratio = 0.2
fig_freq = 10000

show_fig = True
