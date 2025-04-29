from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.structure import SoftmaxLayer
from pybrain3.structure import TanhLayer
net = buildNetwork(2, 3, 1, hiddenclass=TanhLayer, outputclass=SoftmaxLayer, bias=True)
net.activate((2, 3))