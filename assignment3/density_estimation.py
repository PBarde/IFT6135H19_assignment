#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch 
import matplotlib.pyplot as plt

import samplers
from keras.models import Sequential as sq
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import optimizers
import keras.backend as K

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))


############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
EPOCHS = 1000
#Model loss function
def D_Loss(y_true, y_pred):
    # Sample from the gaussian distribution.
    D_x = y_pred[:512]
    D_y = y_pred[512:]

    #Calculate and print the loss
    loss = -(K.mean(K.log(1 - D_y)) + K.mean(K.log(D_x)))

    loss = K.print_tensor(loss)

    return loss

# Create model
discriminator = sq()
discriminator.add(Dense(units=128, activation='relu', input_dim=1))
discriminator.add(Dense(units=128, activation='relu'))
discriminator.add(Dense(units=128, activation='relu'))
discriminator.add(Dense(units=1, activation='sigmoid'))

discriminator.compile(loss=D_Loss,
                      optimizer=Adam(lr=0.00005))

# Make target dummy for Keras
target_dummy = np.zeros(512 * 2)

#Create samplers
gaussian_dist_gen = samplers.distribution3()
mystery_dist_gen = samplers.distribution4(batch_size=512)

for _ in range(EPOCHS):

    #Sample the distributions
    x = next(mystery_dist_gen)
    y = next(gaussian_dist_gen)

    #Train
    discriminator.train_on_batch(np.concatenate((x, y)), target_dummy)

############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density

r = discriminator.predict(xx) # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')

gaussian_dist_gen = samplers.distribution3(batch_size=1000)
f0 = N(xx)
D_x = discriminator.predict(xx)

estimate = (f0 * D_x[:,0])/(1 - D_x[:,0]) # estimate the density of distribution4 (on xx) using the discriminator;
                                # replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')















