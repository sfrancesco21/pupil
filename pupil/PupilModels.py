# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 18:18:32 2022

@author: franc
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split

#import tensorflow_probability as tfp

pupil_ph = np.sin(0.1*np.arange(5120))
pupil_ph = np.reshape(pupil_ph, (1, 5120, 1))
stimuli = np.zeros((1, 5120, 1))
for i in np.arange(0, 5120, 32):
    stimuli[0, i, 0] = np.random.normal(1, 0.1)

T = 5120

class mean_learning():
    #lr -> (1)
    #stimuli -> (T, 1)
    def __init__(self, lr, K):
        self.lr = lr
        self.K = K
    def __call__(self, stimuli):
        stimuli = tf.constant(stimuli, dtype = 'float32')
        U = tf.zeros((stimuli.shape[-2]), self.K) #(T, K)
        I = tf.eye(stimuli.shape[-2]) #(T, T) 
        m = 0.5
        for t in range(stimuli.shape[-2]):
            s = stimuli[0, t, 0] 
            if s != 0:
                pe = s-m #(1)
                pe = tf.cast(pe, dtype = 'float32')
                m = m + self.lr*pe #(1)
                mask = I[:, t] #(T, 1)
                U =  U + mask*tf.math.abs(pe) # (T, 1)
                #U = tf.expand_dims(U, axis = -1)
        return U
            

class CognModel():
    
    def __init__(self, model, model_params, n_predictors):
        self.k = tf.Variable(
            initial_value = model_params,
            dtype="float32", 
            trainable = True, 
            name = 'learning_rate')
        self.model = model(self.k, n_predictors)
    def __call__(self, exp_data):
        U = self.model(exp_data) #(T, K)
        return U
  
    
class ConvGamma():
    
    def __init__(self, kernel_size, 
                 gamma_params = [5, 5, 0.1],
                 trainable = [True, True, True]):
        self.kernel_size = kernel_size
        self.shape = tf.Variable(
            initial_value = np.log(gamma_params[0]),
            dtype="float32", 
            trainable = trainable[0], 
            name = 'log_shape')
        self.scale = tf.Variable(
            initial_value = np.log(gamma_params[1]),
            dtype="float32", 
            trainable = trainable[1], 
            name = 'log_scale')
        self.delay = tf.Variable(
            initial_value = np.log(gamma_params[2]),
            dtype="float32", 
            trainable = trainable[2], 
            name = 'log_delay')

    def __call__(self, U):
        
        if len(U.shape) < 2:
            U = tf.reshape(U, (U.shape[0], 1)) # (T, 1) else (T, K)
        def gamma_kernel(shape, scale, delay, size):    
            x = tf.range(0, size, dtype = 'float32')
            x = tf.keras.activations.relu(x-delay)#(size)
            frac = 1/(tf.math.exp(tf.math.lgamma(shape))*tf.math.pow(scale, shape))
            kernel = frac*tf.math.pow(x, shape-1)*tf.math.exp(-x/scale) # (size)
            #kernel = tf.expand_dims(kernel, axis = -1) # (size, 1)
            return kernel
        
        #x = tf.range(0, self.kernel_size, dtype = 'float32')
        #kernel_fun = tfp.distributions.Gamma(concentration = tf.math.exp(self.shape), 
        #                                     rate = tf.math.exp(self.scale), 
        #                                     force_probs_to_zero_outside_support=True)
        kernel = gamma_kernel(tf.math.exp(self.shape), 
                              tf.math.exp(self.scale), 
                              tf.math.exp(self.delay), 
                              self.kernel_size) #(size, 1)
        #nonneg = tf.keras.activations.relu(x-tf.math.exp(self.delay))
        #kernel = kernel_fun.prob(nonneg)
        def conv_1D(U, kernel):
            kernel = tf.reverse(kernel, axis = [0])
            ulen = U.shape[0] #T
            K = U.shape[1]
            klen = kernel.shape[0] #size
            U = tf.concat([tf.zeros((kernel.shape[0]-1, K)), U], axis = 0) #(T+size-1, 1)
            X = tf.zeros((1,K))
            kernel = tf.expand_dims(kernel, -1)
            for l in range(ulen):
                j = U[l:l+klen, :]
                cx = tf.math.reduce_sum(tf.math.multiply(kernel, j), 
                                        axis = 0,
                                        keepdims = True)
                X  = tf.concat([X, cx], axis = 0)
            return X[:-1,:] #(T, K)
        
        X = conv_1D(U, kernel)
            
        return X
    

class GLM_AR():
    
    def __init__(self, n_predictors, ar = 1):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape = (n_predictors, 1)),
            dtype="float32", 
            trainable = True, 
            name = 'weights')
        self.a = tf.Variable(
            initial_value = w_init(shape = (ar, 1)),
            dtype="float32", 
            trainable = True, 
            name = 'AR_coefficients')
        self.ar = ar
        self.r = tf.Variable(
            initial_value = w_init(shape = (1,1)),
            dtype="float32", 
            trainable = True, 
            name = 'r')
        
    def __call__(self, X, y):
        z = tf.matmul(X, self.w) # (T, 1)    
        e = y - z # (T, 1)
        e = tf.reshape(e, (e.shape[1], e.shape[2]))
        err = tf.zeros(e.shape) #(T, 1)
        if self.ar>0:
            for n in range(self.ar):
                shift = tf.concat([self.r*tf.ones((n+1,1)), e[:-n-1,:]], 
                                  axis = 0) #T,1
                err = tf.concat([err, shift], axis = -1) #(T, ar+1) 
        
            apad = tf.concat([tf.zeros((1, 1)), self.a], axis = 0) #(ar+1, 1)
            sumerr = tf.matmul(err, apad) #(T, 1)
            z = z + sumerr #(T, 1)
        #z = tf.squeeze(z, axis = -1)
        
        return z
    
##########


def training_loop(stim, y, epochs):
    mse_loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(clipvalue = 0.5)
    
    cognitive_model = CognModel(model = mean_learning, 
                                model_params = 0.5, 
                                n_predictors = 1)
    gamma_convolution = ConvGamma(kernel_size = 200)     
    glm = GLM_AR(n_predictors = 1, ar = 1) 
    totrain = [cognitive_model.k, 
               gamma_convolution.shape, 
               gamma_convolution.scale,
               gamma_convolution.delay, 
               glm.w, glm.a, glm.r
               ]
    for e in range(epochs):
        
        with tf.GradientTape() as tape:
            tape.watch(totrain)
            U = cognitive_model(stim)
            X = gamma_convolution(U)
            z = glm(X, y)
            
            loss = mse_loss(y, z)
            
        gradients = tape.gradient(loss, totrain)
        optimizer.apply_gradients(zip(gradients, totrain))
        print('MSE loss: ', loss)
        
        return U, X, z
    
U, X, z = training_loop(stimuli, pupil_ph, 10)

#####

class FullModelSpace(): 
    def __init__(self, models,
                 optimizer,
                 model_params, 
                 kernel_size, 
                 n_predictors, 
                 ar):
        self.models = models
        self.model_params = model_params
        self.cog_models = []
        self.gamma_convolutions = []
        self.glms = []
        self.train_losses = []
        self.test_losses = []
        for m in range(len(models)):
            self.cog_models.append(CognModel(models[m], 
                                             model_params[m], 
                                             n_predictors[m]))
            self.gamma_convolutions.append(ConvGamma(kernel_size[m]))
            self.glms.append(GLM_AR(n_predictors[m], ar[m]))
            self.train_losses.append([])
            self.test_losses.append([])
        self.optimizer = optimizer
        
    def make_masks(self, n_trials, trial_len, 
               k_fold_CV = False, k = 20, 
               test_split = 0.2):
        train_mask = []
        test_mask = []
        if k_fold_CV:
            splitter = KFold(n_splits = k, 
                             shuffle = True)
            splits = splitter.split(np.arange(n_trials)[1:])
            for split in splits:
                train_idx = split[0] + 1
                mask = np.zeros((1, n_trials))
                mask[0, train_idx] = 1 # never test on trial 1?
                mask[0,0] = 1
                mask = tf.repeat(mask, trial_len, axis=1)
                train_mask.append(mask)
                test_mask.append(1-mask)
        else:
            train_idx, _ = train_test_split(np.arange(n_trials)[1:], 
                                            test_size = test_split)
            mask = np.zeros((1, n_trials))
            mask[0, train_idx] = 1
            mask = tf.repeat(mask, trial_len, axis=1)
            train_mask.append(mask)
            test_mask.append(1-mask)
        self.train_mask = train_mask
        self.test_mask = test_mask
        
    def fitall(self, stim, y, 
               patience = 3, 
               threshold = 0.001):
        
        mse_loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE)
        
        def mse_loss_masked(y_true, y_pred, mask):
            mask = tf.cast(mask, dtype = 'float32')
            mask /= tf.math.reduce_mean(mask)
            loss_unmasked = mse_loss(y_true, y_pred)
            loss = tf.math.reduce_mean(mask*loss_unmasked)
            return loss
        
        for m in range(len(self.cog_models)):
            totrain = [self.cog_models[m].k, 
               self.gamma_convolutions[m].shape, 
               self.gamma_convolutions[m].scale,
               self.gamma_convolutions[m].delay, 
               self.glms[m].w, self.glms[m].a, self.glms[m].r
               ]
            for s in range(len(self.train_mask)):
                loss_old = 122.
                step = 0
                count = 0
                while count < patience:
                    with tf.GradientTape() as tape:
                        tape.watch(totrain)
                        U = self.cog_models[m](stim)
                        X = self.gamma_convolutions[m](U)
                        z = self.glms[m](X, y)
                        
                        loss_new = mse_loss_masked(y, z, self.train_mask[s])
                
                    gradients = tape.gradient(loss_new, totrain)
                    self.optimizer.apply_gradients(zip(gradients, totrain))
                    
                    if loss_old-loss_new < threshold:
                        count +=1
                    else:
                        count = 0
                    
                    step += 1
                    print('Step', step, ' MSE loss: ', loss_new.numpy, 
                          'Count :', count)
                    loss_old = loss_new
                
                #eval
                U = self.cog_models[m](stim)
                X = self.gamma_convolutions[m](U)
                z = self.glms[m](X, y)
                
                self.train_losses[m].append(mse_loss_masked(y, z, 
                                                            self.train_mask[s]))
                self.test_losses[m].append(mse_loss_masked(y, z, 
                                                            self.test_mask[s]))
        
        return self.train_losses, self.test_losses
    
#####

model = FullModelSpace([mean_learning], tf.keras.optimizers.Adam(), 
                       [0.3], [200], [1], [1])

model.make_masks(160, 32, k_fold_CV=True)


model.fitall(stimuli, pupil_ph, threshold = 0.1)
