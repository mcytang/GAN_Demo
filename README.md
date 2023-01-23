# GAN_Demo

This is a demonstratory exercise in using the GAN framework, as proposed by Goodfellow in 2014 (https://arxiv.org/abs/1406.2661).

## Purpose

To create a pseudo random number generator using the GAN framework with least squares loss criterion, 
aimed to mimic a chosen probability distribution.

## Use 

To run the demo excecute

    python main.py
    
A generator and discriminator will be saved to /TrainedModels. These models can be tested by calling

    python test.py "MODEL_NAME_HERE"
    
For example, to test the example network provided, call

    python test.py "22_01_2023-22_09"
    
 This will create a pyplot figure to compare the generated distribution (orange line) 
 to the target distribution (indicated by blue bars).
 
 To train your own model, parameters can be sat within Hyperparameters.py.
 
 ## About 
 
 A probability distribution can be reasonably well 
 approximated by simply approximating a function, e.g. $e^{-x^2}$ 
 for a Gaussian or normal distribution, assuming we already have access
 to a uniform random variable. We use uniformly sampled numbers as
 seeds for our generative model. As such, our generator has a lightweight 
 architecture, consisting of a small number of linear (or fully connected) layers.
 
 The discriminator is similary simple, and the main difficulty is balancing
 the width and depth of the generator and discriminator to achieve a 
 meaningful training procedure.
 
 To avoid the common _vanishing gradient_ problem, we train using the 
 least squares criterion proposed by Mao _et al_ in 2016 
 (https://arxiv.org/abs/1611.04076). Moreover, we use specral normalisation
 in the discriminator.
 
 
 
 ## To do
 
 - Test for effective parameters with conventional GAN loss
 - Add GPU functionality
