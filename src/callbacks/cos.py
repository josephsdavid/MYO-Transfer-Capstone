import numpy as np
import matplotlib.pyplot as plt
import math


x = np.arange(1000)
mi = 5
ma = 500
gamma = 0.85
t = 10
out = np.zeros(1000)
for i in range(x.shape[0]):
    out[i] = mi + (ma-mi)*(1+math.cos(math.pi*x[i]/t))/2
    if out[i] == mi:
        ma *= gamma
plt.plot(out)
plt.show()


