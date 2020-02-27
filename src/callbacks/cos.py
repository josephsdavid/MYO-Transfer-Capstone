import numpy as np
import matplotlib.pyplot as plt
import math


x = np.arange(1000)
mi = 1e-5
ma = 1e-3
gamma = 0.9
rep = 50
t = 50
out = np.zeros(1050)
for i in range(x.shape[0]):
    if i>4:
        out[i] = mi + (ma-mi)*(1+math.cos(math.pi*x[i%(rep+5)]/t))/2
    else:
        out[i] = ma
    if out[i] == mi:
        ma *= gamma
plt.plot(out)
plt.show()


