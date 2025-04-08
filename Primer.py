from pybrain3.tools.shortcuts import buildNetwork

net = buildNetwork(2, 3, 1) # вход, скрытый слой, выход

y = net.activate([2, 1])
print("y = ", y)

#TODO Исправить библиотеку!!!