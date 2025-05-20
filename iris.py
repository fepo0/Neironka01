from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
print("Тип массива data: {}".format(type(iris_dataset['data'])))
print("Форма массива data: {}".format(iris_dataset['data'].shape))
print("Цель: {}".format(iris_dataset['target']))
print("Названия ответов: {}".format(iris_dataset['target_names']))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Названия признаков: \n{}".format(iris_dataset['feature_names']))
print("Расположение файла: \n{}".format(iris_dataset['filename']))