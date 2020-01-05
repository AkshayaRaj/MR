#! /usr/bin/env python

import numpy as np
import math
import scipy.stats
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def likelihood(m):
    """ 
    Calculate the likelihood that your friend is at place m
    args: m - - place [x,y]
    """

    x_0 = np.array([12, 4]) # tower 0
    x_1 = np.array([5, 7]) # tower 1
    d_0 = 3.9
    d_1 = 4.5
    var_0 = 1
    var_1 = 1.5 

    # calculate the expected distance measurements 
    d_0_hat  = np.sqrt(np.sum(np.square(m - x_0)))
    d_1_hat  = np.sqrt(np.sum(np.square(m - x_1)))

    pdf_0 = scipy.stats.norm.pdf(d_0, d_0_hat, np.sqrt(var_0))
    pdf_1 = scipy.stats.norm.pdf(d_1, d_1_hat, np.sqrt(var_1))

    return pdf_0 * pdf_1

# locations of interest 
m_0 = np.array([10, 8]) # university
m_1 = np.array([6, 3])  # home
x_0 = np.array([12, 4]) # tower 0
x_1 = np.array([5, 7])  # tower 1

# mesh grid for plotting
x = np.arange(3.0, 15.0, 0.5)
y = np.arange(-5.0, 15.0, 0.5 )
X, Y = np.meshgrid(x, y)

# calculate the likelihood for each position
Z = np.array([likelihood(np.array([x, y])) for x, y in zip(X.flatten(), Y.flatten())])
Z = Z.reshape(X.shape) # X.shape = Y.shape

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cm.coolwarm,alpha=0.5)
ax.scatter(m_0[0],m_0[1],likelihood(m_0),c='g',marker='o',s=100)
ax.scatter(m_1[0],m_1[1],likelihood(m_1),c='r',marker='o',s=100)
ax.scatter(x_0[0],x_0[1],likelihood(x_0),c='g',marker='^',s=100)
ax.scatter(x_1[0],x_1[1],likelihood(x_1),c='r',marker='^',s=100)

plt.show()