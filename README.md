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

    python test.py "22_01_2023-21_12"
    
 This will create a pyplot figure to compare the generated distribution (orange line) 
 to the target distribution (indicated by blue bars).
 
 To train your own model, parameters can be sat within Hyperparameters.py.
 
 ## About 
 
 Given that a probability distribution can be reasonably well 
 approximated by simply approximating a function, e.g. $e^{-x^2}$ for a Gaussian or normal distribution.
 
 ## To do
 
 - Test for effective parameters with conventional GAN loss
 - Add GPU functionality
