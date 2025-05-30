import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker='.')
plt.savefig("graph.png")