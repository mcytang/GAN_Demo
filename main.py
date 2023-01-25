from datetime import datetime
import os
from time import time
from torch import manual_seed

from model import model as M 
from utils.train import GANtrain
import utils.helpers  as h
import Hyperparameters as P

manual_seed(P.seed)

G = M.Generator(P.target_distribution, P.mu, P.std, P.Nsample, P.depth, P.width)
D = M.Discriminator(P.Nsample, P.D_inChannels, P.D_depth, loss = P.loss)

loss_func = h.get_loss(P.loss)


train = GANtrain(
    P.G_lr, 
    P.D_lr,
    P.Nepochs, 
    P.batchSize, 
    P.drop_rate, 
    P.drop_ratio, 
    P.show_fig,
    P.fig_freq,
    P.save_freq
)

model_name = datetime.now().strftime('%d_%m_%Y-%H_%M')
path = r'TrainedModels/{}'.format(model_name)
if not os.path.exists(path): os.mkdir(path)

h.save_parameters(model_name, P)

t0 = time()
G, D = train(model_name, G, D, loss_func, P.mu, P.std)
t = time() - t0

h.save_model(model_name, G, D, P.show_fig)
h.save_time(model_name, t)