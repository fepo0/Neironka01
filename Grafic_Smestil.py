import matplotlib.pylab as plt
import numpy as np

x = np.arange(-8, 9, 0.1)

b1 = -2
b2 = 0
b3 = 2

l1 = "Смещение b = -2"
l2 = "Смещение b = 0"
l3 = "Смещение b = 2"

for b, l in [(b1, l1), (b2, l2), (b3, l3)]:
    f = (1 / (1 + np.exp((-x + b) * 1)))
    plt.plot(x, f, label=l)

plt.xlabel("x")
plt.ylabel("Y = f(x)")
plt.legend(loc=4)
plt.savefig("Grafic_Smestil.png")