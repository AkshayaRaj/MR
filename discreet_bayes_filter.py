#! /usr/bin/env python

# check slide 4 of lecture on discreet bayes filter 

import numpy as np 
import matplotlib.pyplot as plt



def bayes_filter(bel,d):
    bel_prime = np.zeros(bel.shape)
    # given a state belief bel and control d = <f ,b > compute the resulting state belief bel_prime
    if d == 'f':
        for x in range(20): # 0..19
            if x >= 2:
                bel2 = bel[x-2]
            else:
                bel2 = 0
            if x >= 1:
                bel1 = bel[x-1]
            else:
                bel1 = 0
            bel0 = bel[x]
            if x < 19:
                bel_prime[x] = bel0*0.25 + bel1*0.5 + bel2*0.25
            elif x == 19: # last cell
                bel_prime[x] = bel0*1 + bel1*0.75 + bel2*0.25
    if d == 'b':
        for x in range(20):
            if x <= 17:
                bel2 = bel[x+2]
            else:
                bel2 = 0
            if x <= 18:
                bel1 = bel[x+1]
            else:
                bel1 = 0
            bel0 = bel[x]
            if x > 0:
                bel_prime[x] = bel0*0.25 + bel1*0.5 + bel2*0.25
            elif x == 0:
                bel_prime[x] = bel0*1 + bel1*0.75 + bel2*0.25

    return bel_prime
    


def plot_historgram(bel):
    plt.cla() 
    plt.bar(range(0,bel.shape[0]),bel,width=1.0)
    plt.axis([0,bel.shape[0]-1,0,1])
    plt.draw()
    plt.pause(1)

def main():
    # initial belief:
    bel = np.hstack ((np.zeros(10), 1, np.zeros(9)))

    plt.figure()
    plt.ion()
    plt.show()

    for i in range(9):
        plot_historgram(bel)
        bel = bayes_filter(bel,'f')
        print(np.sum(bel))

    for i in range(3):
        plot_historgram(bel)
        bel = bayes_filter(bel,'b')
        print(np.sum(bel))
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
