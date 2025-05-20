import pandas as pd

data = {'Имя': ["Дима", "Анна", "Петр", "Вика"],
        'Город': ["Москва", "Курск", "Псков", "Воронеж"],
        'Возраст': [24, 13, 53, 33]}
data_pandas = pd.DataFrame(data)
print(data_pandas)

