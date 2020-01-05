#! /usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt

plt.gca().set_aspect('equal')

def diffdrive(x,y,theta,v_l,v_r,t,l):

    # straight line
    if(v_r == v_l):
        theta_n = theta
        x_n = x + v_r * np.cos(theta) * t
        y_n = y + v_r * np.sin(theta) * t
        return x_n, y_n , theta_n

    R = l/2 * ((v_r + v_l) / (v_r - v_l))
    
    # calculate ICC_x , ICC_y
    ICC_x = x - (R * math.sin(theta))
    ICC_y = y + (R * math.cos(theta))
    p_1 = np.array([x - ICC_x , y - ICC_y, theta ])
    w = (v_r - v_l) / l
    T = np.array(
        [[np.cos(w*t), -np.sin(w*t), 0],
        [np.sin(w*t), np.cos(w*t), 0],
        [0 , 0, 1]
        ])
    p_2 = np.dot(T,p_1) + np.array([ICC_x, ICC_y, w*t])
    x_n = p_2[0]
    y_n = p_2[1]
    theta_n = p_2[2]
    return x_n,y_n,theta_n

plt.quiver(1.5, 2.0, np.cos(math.pi/2), np.sin(math.pi/2))
c1_x, c1_y, c1_theta = diffdrive(1.5, 2.0, math.pi/2, 0.3, 0.3, 3, 0.5) 
plt.quiver(c1_x, c1_y, np.cos(c1_theta), np.sin(c1_theta))
c2_x, c2_y, c2_theta = diffdrive(c1_x, c1_y, c1_theta, 0.1, -0.1, 1, 0.5) 
plt.quiver(c2_x, c2_y, np.cos(c2_theta), np.sin(c2_theta))
c3_x, c3_y, c3_theta = diffdrive(c2_x, c2_y, c2_theta, 0.2, 0, 2, 0.5) 
plt.quiver(c3_x, c3_y, np.cos(c3_theta), np.sin(c3_theta))

print(c1_x, c1_y, c1_theta)
print(c2_x, c2_y, c2_theta)
print(c3_x, c3_y, c3_theta)

plt.xlim([0.5, 2.5])
plt.ylim([1.5, 3.5])

plt.show()