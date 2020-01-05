#! /usr/bin/env python
import numpy as np 
import math

import sampler as sampler
import matplotlib.pyplot as plt

def sample_normal(mu,sigma):
    #return sampler.sample_normal_distribution(mu,sigma)
    return sampler.sample_box_mulller(mu,sigma)

def sample_motion_model(u,x,alpha):
    # u = delta_r1, trans, delta_r2
    # x = x, y, theta
    # aplha = [a1, a2, a3, a4]
    delta_hat_r1 = u[0] + sample_normal(0, alpha[0] * abs(u[0]) + alpha[1] * abs(u[1]) )
    delta_hat_trans = u[1] + sample_normal(0, alpha[2] * abs(u[1]) + alpha[3] * (abs(u[0]) + abs(u[2])))
    delta_hat_r2 = u[2] + sample_normal(0, alpha[0] * abs(u[2]) + alpha[1] * abs(u[1]) )

    x_new = x[0] + delta_hat_trans * np.cos(x[2] + delta_hat_r1)
    y_new = x[1] + delta_hat_trans * np.sin(x[2] + delta_hat_r1)
    theta_new = x[2] + delta_hat_r1 + delta_hat_r2

    return np.array([x_new, y_new, theta_new])

def main():
    x = [2.0, 4.0, 0.0]
    u = [math.pi/2, 1.0, 0.0]
    alpha = [0.1, 0.1, 0.01, 0.01]
    n_samples  = 5000
    state = np.zeros([n_samples,3])
    for i in range(n_samples):
        state[i,:] = sample_motion_model(u,x,alpha)
    #print state.shape
    plt.plot(x[0], x[1] ,"bo")
    plt.plot(state[:,0], state[:,1],"r,")

    plt.axes().set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    main()