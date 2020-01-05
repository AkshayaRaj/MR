#! /usr/bin/env python

import numpy as np 
import scipy.stats
import random
import timeit

# In the first function, generate the normal distributed samples by summing up 12 uniform distributed samples

# mu : mean of the normal distribution
# sigma: std_dev of the normal distribution
def sample_normal_distribution(mu,sigma):
    x = 0.5 * np.sum(np.random.uniform(-sigma,sigma,12))
    return mu + x # why add mu ? 

def sample_rejection_sampling(mu,sigma):
    interval = 5 * sigma # why 5 times sigma ? 
    # max value of pdf will occur at sigma ~: 
    max = scipy.stats.norm(mu,sigma).pdf(sigma)

    while True:
        x = np.random.uniform(mu-interval, mu+interval)
        y = np.random.uniform(0,max)
        if scipy.stats.norm(mu,sigma).pdf(x)  >= y:
            break
    return x

def sample_box_mulller(mu,sigma):
    u1 , u2 = np.random.uniform(0,1,2)
    x = np.cos(2 * np.pi * u1) * np.sqrt(-2 * np.log(u2))
    return mu + sigma * x

def evaluate_timing(mu,sigma,n_samples,function):
    tick = timeit.default_timer()
    for i in range(n_samples):
        function(mu,sigma)
    tock = timeit.default_timer()
    dt  = (tock - tick ) / n_samples * 1e6
    print ("%30s : %.3f micro seconds" %(function.__name__,dt))


def main():
    mu = 0 
    sigma = 1
    functions = [sample_normal_distribution,sample_rejection_sampling,sample_box_mulller,np.random.normal]
    for fnc in functions:
        evaluate_timing(mu, sigma, 1000,fnc)
    

if __name__=='__main__':
    main()

x1 = sample_normal_distribution(5,3)
x2 = sample_rejection_sampling(5,3)
x3 = sample_box_mulller(5,3)

