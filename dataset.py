from matplotlib import pyplot as plt
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised import BackpropTrainer
from pybrain3.tools.shortcuts import buildNetwork

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

trainer = BackpropTrainer(net, ds)
print(trainer.train())

trainer.trainUntilConvergence()

trainer = BackpropTrainer(net)
trnerr, valerr = trainer.trainUntilConvergence(dataset=ds, maxEpochs=100)
plt.plot(trnerr, 'b', valerr, 'r')
plt.savefig("datasetZavisimost.png")