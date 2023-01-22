# seed 
seed = 8

# Generator parameters
target_distribution = 'Gaussian'
mu = 0
std = 1
Nsample = 2 ** 3
Nseed = 1
inChannels = 4
depth = 8

# Discriminator

D_inChannels = 256
D_depth = 11

#Training
lr = 1e-3
Nepochs = 45000
batchSize = 32
drop_rate = 30000
drop_ratio = 0.2
fig_freq = 5000

show_fig = True
