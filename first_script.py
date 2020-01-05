#! /usr/bin/env python

import math
import matplotlib.pyplot as plt
import numpy as np

def calculate(val):
      return math.cos(val) * math.exp(val)

x = np.arange(-2 * np.pi, 2 * np.pi, 0.1);
f = np.vectorize(calculate)

plt.plot(x,f(x))
plt.show()

