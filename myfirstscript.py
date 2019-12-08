import math
import matplotlib.pyplot as plt
import numpy as np

def calc(x):
  return math.cos(x)*math.exp(x)

f = np.vectorize(calc)
x = np.arange(-2*np.pi,2*np.pi,0.1);

plt.plot(x,f(x))
plt.show()
  

