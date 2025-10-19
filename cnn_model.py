import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.utils import to_categorical
import matplotlib.pyplot as plt

np.random.seed(123)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Создание модели
model = Sequential()

# Первый сверточный слой
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))

# Второй сверточный слой
model.add(Conv2D(32, kernel_size=3, activation='relu'))

# Вектор для полносвязной сети
model.add(Flatten())

# Однослойный персептрон
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)
print(history.history)

# Построение графика точности предсказания
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

# todo дописать + построение графика потерь
