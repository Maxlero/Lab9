"""
    Автор: Орел Максим
    Группа: КБ-161
    Вариант: 11
    Дата создания: 2/05/2018
    Python Version: 3.6
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d

# useful to understand http://mathprofi.ru/metod_naimenshih_kvadratov.html

x = [4.08, 4.42, 2.52, -0.08, 2.14, 3.36, 7.35, 5.00]
y = [18.31, 21.85, 16.93, -8.23, 10.90, 17.18, 36.45, 24.11]


def build_points(x_array, y_array):
    for i in range(0, len(x_array)):
        plt.scatter(x_array[i], y_array[i])


def linear_function(x_array, y_array):
    # solving next system
    # a*x^2 + b*x - x*y = 0
    # a*x + b - y = 0
    sums = [0, 0, 0, 0]
    for i in range(len(x_array)):
        sums[0] += x_array[i] * x_array[i]
        sums[1] += x_array[i]
        sums[2] += x_array[i] * y_array[i]
        sums[3] += y_array[i]

    left = np.array([[sums[0], sums[1]], [sums[1], len(x_array)]])
    right = np.array([sums[2], sums[3]])
    a, b = np.linalg.solve(left, right)

    deviation = 0
    for i in range(len(x_array)):
        deviation += (y_array[i] - a * x_array[i] - b) ** 2
    print('Отклонение для y = a*x + b:', deviation)

    points = np.linspace(min(x_array) - 0.228, max(x_array) + 0.228)
    plt.plot(points, a * points + b, label='y = a*x + b')


def hyperbolic_function(x_array, y_array):
    # solving next system
    # a/x^2 + b/x - y/x = 0
    # a/x + b - y = 0
    sums = [0, 0, 0, 0]
    for i in range(len(x_array)):
        sums[0] += 1 / (x_array[i] * x_array[i])
        sums[1] += 1 / x_array[i]
        sums[2] += y_array[i] / x_array[i]
        sums[3] += y_array[i]

    left = np.array([[sums[0], sums[1]], [sums[1], len(x_array)]])
    right = np.array([sums[2], sums[3]])
    a, b = np.linalg.solve(left, right)

    deviation = 0
    for i in range(len(x_array)):
        deviation += (y_array[i] - a / x_array[i] - b) ** 2
    print('Отклонение для y = a/x + b:', deviation)

    points = np.linspace(min(x_array) - 0.228, max(x_array) + 0.228)
    plt.plot(points, a / points + b, label='y = a/x + b')


def logarithmic_function(x_array, y_array):
    # solving next system
    # a*ln(x)^2 + b*ln(x) - y*ln(x) = 0
    # a*ln(x) + b - y = 0
    sums = [0, 0, 0, 0]
    for i in range(len(x_array)):
        if x_array[i] < 0:
            continue
        sums[0] += np.log(x_array[i]) * np.log(x_array[i])
        sums[1] += np.log(x_array[i])
        sums[2] += y_array[i] * np.log(x_array[i])
        sums[3] += y_array[i]

    left = np.array([[sums[0], sums[1]], [sums[1], len(x_array)]])
    right = np.array([sums[2], sums[3]])
    a, b = np.linalg.solve(left, right)

    deviation = 0
    for i in range(len(x_array)):
        if x_array[i] < 0:
            continue
        deviation += (y_array[i] - a * np.log(x_array[i]) - b) ** 2
    print('Отклонение для y = a*ln(x) + b:', deviation)

    points = np.linspace(0.1, max(x_array) + 0.228)
    plt.plot(points, a * np.log(points) + b, label='y = a*ln(x) + b')


def polynomial(x_array, y_array):
    # solving next system
    # a*x^4 + b*x^3 + c*x^2 - x^2*y = 0
    # a*x^3 + b*x^2 + c*x - x*y = 0
    # a*x^2 + b*x + c - y = 0
    sums = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(x_array)):
        sums[0] += x_array[i] ** 4
        sums[1] += x_array[i] ** 3
        sums[2] += x_array[i] ** 2
        sums[3] += x_array[i]
        sums[4] += x_array[i] ** 2 * y_array[i]
        sums[5] += x_array[i] * y_array[i]
        sums[6] += y_array[i]

    left = np.array([[sums[0], sums[1], sums[2]], [sums[1], sums[2], sums[3]], [sums[2], sums[3], len(x_array)]])
    right = np.array([sums[4], sums[5], sums[6]])
    a, b, c = np.linalg.solve(left, right)

    deviation = 0
    for i in range(len(x_array)):
        deviation += (y_array[i] - a * x_array[i] ** 2 - b * x_array[i] - c) ** 2
    print('Отклонение для y = a*x*x + b*x + c:', deviation)

    points = np.linspace(min(x_array) - 0.228, max(x_array) + 0.228)
    plt.plot(points, a * points ** 2 + b * points + c, label='y = a*x*x + b*x + c')


def exponential_function(x_array, y_array):
    # solving next system
    # a*x^2 + B*x - x*ln(y) = 0
    # a*x + B - ln(y) = 0
    sums = [0, 0, 0, 0]
    for i in range(len(x_array)):
        if x_array[i] < 0:
            continue
        sums[0] += x_array[i] * x_array[i]
        sums[1] += x_array[i]
        sums[2] += x_array[i] * np.log(y_array[i])
        sums[3] += np.log(y_array[i])

    left = np.array([[sums[0], sums[1]], [sums[1], len(x_array)]])
    right = np.array([sums[2], sums[3]])
    a, B = np.linalg.solve(left, right)
    # b = e^B

    deviation = 0
    for i in range(len(x_array)):
        deviation += (y_array[i] - np.exp(B) * np.exp(x_array[i] * a)) ** 2
    print('Отклонение для y = b*e^(a*x):', deviation)

    points = np.linspace(0.1, max(x_array) + 0.228)
    plt.plot(points, np.exp(B) * np.exp(points * a), label='y = b*e^(a*x)')


if __name__ == "__main__":
    try:
        # Построим звезд сегодняшней программы в стройный рядок и начнем наше мероприятие
        build_points(x, y)

        # Бал начнет простой и прямолинейный человек - Лео
        # 1) y = a*x + b
        linear_function(x, y)

        # Встречайте мужчину без майки, который не прочь воспользоваться химией для набора массы
        # 2) y = a/x + b
        hyperbolic_function(x, y)

        # А далее покажет на что способен юный шотландец - Джон Непер
        # 3) y = a*ln(x) + b
        logarithmic_function(x, y)

        # И наконец гвоздь программы - мисс Полина
        # 4) y = a*x*x + b*x + c
        polynomial(x, y)

        # Завершает наше мероприятие молодой и не женатый сын Эйлера - Эйлер
        # 5) y = b*e^(a*x)
        exponential_function(x, y)

        # Тетя Таня разрешила Леонарду расписать тут кисточки и карандаши
        plt.grid()
        plt.legend()
        plt.show()

    except Exception as e:
        print(e)
