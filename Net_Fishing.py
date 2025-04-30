import pickle
import matplotlib.pylab as plt
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer

# 4 входа, 1 выход
ds = SupervisedDataSet(4, 1)
# Скорость ветра, Перепад давления, Облачность, Перепад температуры, Активность рыбы(ответ)
ds.addSample([2, 3, 80, 1], [5])

ds = SupervisedDataSet(4, 1)
ds.addSample([2, 3, 80, 1], [5])
ds.addSample([5, 5, 50, 2], [4])
ds.addSample([10, 7, 40, 3], [3])
ds.addSample([15, 9, 20, 4], [2])
ds.addSample([20, 11, 10, 5], [1])

net = buildNetwork(4, 3, 1, bias=True)

trainer = BackpropTrainer(net, dataset=ds, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.01)
trnerr, valerr = trainer.trainUntilConvergence()

plt.plot(trnerr, 'b',  valerr, 'r')
plt.savefig("Net_FishingAnalyzis.png")

fileObject = open('Net_Fishing.txt', 'wb')
pickle.dump(net, fileObject)
fileObject.close()