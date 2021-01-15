import numpy as np
import matplotlib.pyplot as plt

fname = 'b1422_ratios_1.txt'
r1 = np.loadtxt(fname)
fname = 'b1422_ratios_2.txt'
r2 = np.loadtxt(fname)
fname = 'b1422_ratios_noapprox.txt'
r3 = np.loadtxt(fname)

idx = 1
bins = np.linspace(0.2, 1.3, 10)
plt.hist(r1[:,idx], bins=bins, density=True, color='k', histtype='step')
#plt.hist(r2[:,idx], bins=bins, density=True, color='r', histtype='step')
plt.hist(r3[:,idx], bins=bins, density=True, color='m', histtype='step')
print(r1[:,idx])

plt.show()
