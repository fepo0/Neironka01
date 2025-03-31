import matplotlib.pylab as plt
import numpy as np

x = np.arange(-8, 8, 0.1)

w1 = 0.5
w2 = 1.0
w3 = 2.0

l1 = "Вес w = 0.5"
l2 = "Вес w = 1.0"
l3 = "Вес w = 2.0"

for w, l in [(w1, l1), (w2, l2), (w3, l3)]:
    f = 1 / (1 + np.exp(-x * w))
    plt.plot(x, f, label=l)

plt.xlabel("x")
plt.ylabel("Y = f(x)")
plt.legend(loc=4)
plt.savefig("Grafic_wesa.png")