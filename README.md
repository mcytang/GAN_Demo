# GAN_Demo

This is a demonstratory exercise in using the GAN framework, as proposed by Goodfellow in 2014 (https://arxiv.org/abs/1406.2661).

## Purpose

To create a pseudo random number generator using the GAN framework with least squares loss criterion, 
aimed to mimic a chosen probability distribution.

## Use 

To run the demo excecute

    python main.py
    
A generator (G) and discriminator (D) will be saved to /TrainedModels. These models can be tested by calling

    python test.py "MODEL_NAME_HERE"
    
For example, to test the example network provided, call

    python test.py "26_01_2023-19_14"
    
 This will create a pyplot figure to compare the generated distribution (orange line) 
 to the target distribution (indicated by blue bars).
 
 To train your own model, parameters can be set within Hyperparameters.py. It is often
 commented that a successful GAN requires the correct balancing of power between G
 and D. In this repo, power can be measured by the number of layers and channels 
 per layer for both G and D. As such, it is easy for the user to run experiments to 
 investigate the training process for poorly balanced models using this repo.
 
 ## About 
 
 A probability distribution can be reasonably well 
 approximated by simply approximating a function, e.g. $e^{-x^2}$ 
 for a Gaussian or normal distribution, assuming we already have access
 to a uniform random variable, _e.g._ torch.rand. Using uniformly sampled numbers as
 seeds for our generative model, the generator G has a fully connected 
 architecture, consisting of a small number of linear (or fully connected) + ReLU layers.
 
 The discriminator D is similary simple, differing only in that the output dimension of 
 each layer decreases by half. 
 
 To avoid the common _vanishing gradient_ problem, we train using the 
 least squares criterion proposed by Mao _et al_ in 2016 
 (https://arxiv.org/abs/1611.04076). This indeed results in a better outcome.
 
 
 
 ## To do

 - Add GPU functionality
 - Adam optimiser proved very poor for this model - Investigate!
 - Improve graphical outputs
 - Find successful parameters to train with classical GAN loss
