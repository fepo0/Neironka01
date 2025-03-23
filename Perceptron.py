import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Загрузка из интернета массива из 150 элементов
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print('Массив')
print(df.to_string())

# Выборка из объекта DF 100 элементов (столбец 4 название цветков) и загрузка его в одномерный массив Y и печать
y = df.iloc[0:100, 4].values
print('Значение четвертого столбца Y - 100')
print(y)

# Преобразование названий
y = np.where(y == 'Iris-setosa', -1, 1)
print('Значение названий цветков  в виде -1 и 1, Y - 100')
print(y)

X = df.iloc[0:100, [0, 2]].values
print('Значение X - 100')
print(X)
print('Конец X')

# Формирование параметров значений для вывода на график
plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='щетинистый')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='разноцветный')

plt.xlabel('длина чашелистика')
plt.ylabel('длина лепестка')
plt.legend(loc='upper left')
plt.savefig("iris_gr.png")

class Perceptron:
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter

    # Подгонка модели под тренировочные данные
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    # Рассчитать чистый вход
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # Вернуть метку класса после еденичного скачка
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

#-------------------------------------

# Тренеровка
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Число случаев ошибочной классификации')
plt.savefig("perceptron_gr.png")

# Визуализация границ
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, color=cmap(idx), marker=markers[idx], label=cl)

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('длина чашелистика [см]')
plt.ylabel('длина лепестка [см]')
plt.legend(loc='upper left')
plt.savefig("decision_regions.png")

#-------------------------------------

class AdaptiveLinearNeuron(object):
    def __init__(self, rate=0.01, niter=10):
        self.rate = rate
        self.niter = niter

    def fit(self, X, y):
        self.weight = np.zeros(1 + X.shape[1])
        self.cost = []
        for i in range(self.niter):
            output = self.net_input(X)
            errors = y - output
            self.weight[1:] += self.rate * X.T.dot(errors)
            self.weight[0] += self.rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

#----------------------------------

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

aln1 = AdaptiveLinearNeuron(0.01, 10).fit(X, y)

ax[0].plot(range(1, len(aln1.cost) + 1), np.log10(aln1.cost), marker='o')
ax[0].set_xlabel('Эпохи')
ax[0].set_ylabel("log(Сумма квадратичных ошибок)")
ax[0].set_title("ADALINE. Темп обучения 0.01")

aln2 = AdaptiveLinearNeuron(0.0001, 10).fit(X, y)

ax[1].plot(range(1, len(aln2.cost) + 1), aln2.cost, marker='o')
ax[1].set_xlabel("Эпохи")
ax[1].set_ylabel("Сумма квадратичных ошибок")
ax[1].set_title("ADALINE. Темп обучения 0.0001")
plt.savefig("ochenka_result.png")

#--------------------------------------

# Стандартизация выборок
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

aln = AdaptiveLinearNeuron(0.01, 10)
aln.fit(X_std, y)

plt.plot(range(1, len(aln.cost) + 1), aln.cost, marker='o')
plt.xlabel("Эпохи")
plt.ylabel("Сумма квадратичной ошибки")
plt.savefig("standart_oshibka.png")

plot_decision_regions(X_std, y, classifier=aln)
plt.title("ADALINE (градиентный спуск)")
plt.xlabel("Длина чашелистника [стандартизованная]")
plt.ylabel("Длина лепестка [стандартизованная]")
plt.legend(loc='upper left')
plt.savefig("standart_reshenia.png")

#------------------------------

i1 = [0.25, 1.1]
R1 = aln.predict(i1)
print("R1 = ", R1)

if (R1 == 1):
    print("R1 = Вид Iris setosa")
else:
    print("R1 = Вид Iris versicolor")