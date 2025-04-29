import matplotlib.pylab as plt
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised import BackpropTrainer
from pybrain3.tools.shortcuts import buildNetwork
import pickle
from pybrain3.tools.xml.networkwriter import NetworkWriter
from pybrain3.tools.xml.networkreader import NetworkReader

net = buildNetwork(2, 3, 1)
y = net.activate([2, 1])

ds = SupervisedDataSet(2, 1) # Двумерный вход и одномерный выход

xorModel = [
    [(0, 0), (0,)],
    [(0, 1), (1,)],
    [(1, 0), (1,)],
    [(1, 1), (0,)],
]

for input, target in xorModel:
    ds.addSample(input, target)

trainer = BackpropTrainer(net)
trnerr, valerr = trainer.trainUntilConvergence(dataset=ds, maxEpochs=100)
plt.plot(trnerr, 'b', valerr, 'r')
plt.savefig("datasetZavisimost.png")

for mod in net.modules:
    print("Module: ", mod.name)

    if mod.paramdim > 0:
        print("--parameters: ", mod.params)

    for conn in net.connections[mod]:
        print("-connection: ", conn.outmod.name)

        if conn.paramdim > 0:
            print("- parameters: ", conn.params)

    if hasattr(net, "recurrentConns"):
        print("Recurrent connections")

        for conn in net.recurrentConns:
            print("-", conn.inmod.name, " to ", conn.outmod.name)

            if conn.paramdim > 0:
                print("- parameters: ", conn.params)

y = net.activate([1, 1])
print("Y1 = ", y)

fileObject = open("MyNet.txt", "wb")
pickle.dump(net, fileObject)
fileObject.close()

fileObject = open("MyNet.txt", "rb")
net2 = pickle.load(fileObject)

y = net2.activate([1, 1])
print("Y2 = ", y)

NetworkWriter.writeToFile(net, "MyNet.xml")
net = NetworkReader.readFrom("MyNet.xml")

