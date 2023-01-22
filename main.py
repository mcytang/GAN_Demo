from torch import manual_seed

from model.model import Generator, Discriminator 
from utils.train import GANtrain
from utils.helpers import save_model
import Hyperparameters as P

manual_seed(P.seed)

G = Generator(P.target_distribution, P.mu, P.std, P.Nseed, P.Nsample, P.inChannels, P.depth)
D = Discriminator(P.Nsample, P.D_inChannels, P.D_depth)

train = GANtrain(
    P.lr, 
    P.Nepochs, 
    P.batchSize, 
    P.drop_rate, 
    P.drop_ratio, 
    P.show_fig,
    P.fig_freq 
)

G, D = train(G, D)

save_model(G, D, P.show_fig)