import random

# Обучающая выборка
num0 = list('111101101101111')
num1 = list('001001001001001')
num2 = list('111001111100111')
num3 = list('111001111001111')
num4 = list('101101111001001')
num5 = list('111100111001111')
num6 = list('111100111101111')
num7 = list('111001001001001')
num8 = list('111101111101111')
num9 = list('111101111001111')

nums = [num0, num1, num2, num3, num4, num5, num6, num7, num8, num9]

tema = 5
n_sensor = 15
weights = [0 for i in range(n_sensor)]

# Является ли изображение числом 5
def perceptron(Sensor):
    b = 7
    s = 0
    for i in range(n_sensor):
        s += int(Sensor[i]) * weights[i]
    if s >= b:
        return True
    else:
        return False

# Уменьшение значения весов (если было другое число, а программа сказала 5)
def decrease(number):
    for i in range(n_sensor):
        if int(number[i]) == 1:
            weights[i] -= 1

# Увеличивание значения весов (если была 5, а программа сказала нет)
def increase(number):
    for i in range(n_sensor):
        if int(number[i]) == 1:
            weights[i] += 1

n = 100000
for i in range(n):
    j = random.randint(0, 9)
    r = perceptron(nums[j])

    if j != tema:
        if r:
            decrease(nums[j])
    else:
        if not r:
            increase(nums[tema])

print(j)
print(weights)

# Проверка программы на обучающей выборке
print("0 это 5? ", perceptron(num0))
print("1 это 5? ", perceptron(num1))
print("2 это 5? ", perceptron(num2))
print("3 это 5? ", perceptron(num3))
print("4 это 5? ", perceptron(num4))
print("5 это 5? ", perceptron(num5))
print("6 это 5? ", perceptron(num6))
print("7 это 5? ", perceptron(num7))
print("8 это 5? ", perceptron(num8))
print("9 это 5? ", perceptron(num9))

num51 = list('111100111000111')
num52 = list('111100010001111')
num53 = list('111100011001111')
num54 = list('110100111001111')
num55 = list('110100111001011')
num56 = list('111100101001111')

print("--------------------")

# Прогон по тестовой выборке
print("Узнал 5 в 5? ", perceptron(num5))
print("Узнал 5 в 51? ", perceptron(num51))
print("Узнал 5 в 52? ", perceptron(num52))
print("Узнал 5 в 53? ", perceptron(num53))
print("Узнал 5 в 54? ", perceptron(num54))
print("Узнал 5 в 55? ", perceptron(num55))
print("Узнал 5 в 56? ", perceptron(num56))
