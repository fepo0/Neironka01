from keras.datasets import mnist
import  matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Тренировачный набор данных ", X_train.shape)
print("Метки тренироваочного набора данных ", y_train.shape)
print("Тестовый набор данных ", X_test.shape)
print("Метки тестового набора данных ", y_test.shape)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[1], cmap=plt.cm.binary)
plt.savefig("cnn.png")