from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Метки тренировочного набора данных после преобразования ", y_train.shape)
print("Метки тестового набора данных после преобразования ", y_test.shape)

y0 = y_train[0]
y1 = y_train[1]
y2 = y_train[0]

print(y0, y1, y2)

