import pickle

fileObject = open("Net_Fishing.txt", "rb")
net2 = pickle.load(fileObject)
fileObject.close()

y = net2.activate([2, 3, 80, 1])
print("Y1 = ", y)

y = net2.activate([10, 7, 40, 3])
print("Y2 = ", y)

y = net2.activate([20, 11, 10, 5])
print("Y3 = ", y)