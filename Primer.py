from pybrain3.tools.shortcuts import buildNetwork

net = buildNetwork(2, 3, 1) # вход, скрытый слой, выход

y = net.activate([2, 1])
print("y = ", y)

a = net['bias']
b = net['in']
c = net['hidden0']
d = net['out']
print(a)
print(b)
print(c)
print(d)
