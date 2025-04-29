import pickle
import matplotlib.pylab as plt
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer

# 4 входа, 1 выход
ds = SupervisedDataSet(4, 1)
# Скорость ветра, Перепад давления, Облачность, Перепад температуры, Активность рыбы(ответ)
ds.addSample([2, 3, 80, 1], [5])

#TODO Дописать обучающий набор данных
#TODO Дописать формирование структуры нейроной сети