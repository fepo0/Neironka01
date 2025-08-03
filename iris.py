import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
print("Тип массива data: {}".format(type(iris_dataset['data'])))
print("Форма массива data: {}".format(iris_dataset['data'].shape))
print("Цель: {}".format(iris_dataset['target']))
print("Названия ответов: {}".format(iris_dataset['target_names']))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Названия признаков: \n{}".format(iris_dataset['feature_names']))
print("Расположение файла: \n{}".format(iris_dataset['filename']))

print("Первые 5 строк массива data:\n{}".format(iris_dataset['data'][:5]))

print("Правильные ответы:\n{}".format(iris_dataset['target']))

# Тренировка -----------
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                    iris_dataset['target'],
                                                    random_state=0)

print("Размерность масива X_train: {}".format(X_train.shape))
print("Размерность масива y_train: {}".format(y_train.shape))
print("Размерность масива X_test: {}".format(X_test.shape))
print("Размерность масива y_test: {}".format(y_test.shape))

# Создание и обучение классификатора
knn = KNeighborsClassifier(n_neighbors=1)
z = knn.fit(X_train, y_train)
print(z)

KNeighborsClassifier(algorithm='auto', leaf_size=30,
                     metric='minkowski', metric_params=None,
                     weights='uniform')
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("Форма массива X_new: {}".format(X_new.shape))
pr = knn.predict(X_new)

print("Метка вида цветка: {}".format(pr))
print("Вид цветка: {}".format(iris_dataset['target_names'][pr[0]]))

df = sb.load_dataset('iris')
sb.set_style("ticks")
sb.pairplot(df, hue='species', diag_kind="kde", kind='scatter', palette="husl")

pr = knn.predict(X_test)
print("Прогноз вида на тестовом наборе:\n {}".format(pr))
print("Точность прогноза на тестовом наборе: {:.2f}".format(np.mean(pr == y_test)))

plt.savefig('matricha_rasseania.png')