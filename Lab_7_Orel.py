"""
    Автор: Орел Максим
    Группа: КБ-161
    Вариант: 11
    Дата создания: 19/04/2018
    Python Version: 3.6
"""
import math
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Constants
accuracy = 0.00001
START_X = 0.2
END_X = 0.8
START_Y = 1
END_Y = 3

x = [0.35, 0.41, 0.47, 0.51, 0.56, 0.64]
y = [2.73951, 2.30080, 1.96864, 1.78776, 1.59502, 1.34310]
point = 0.552


def build_points(x_array, y_array):
    for i in range(0, len(x_array)):
        plt.scatter(x_array[i], y_array[i])


def knowledge_of_maya(x_array, y_array, point):
    return_value = 0

    for i in range(0, len(y_array)):
        temp = 1
        for j in range(0, len(y_array)):
            if i == j:
                continue
            else:
                temp *= (point - x_array[j]) / (x_array[i] - x_array[j])

        return_value += y_array[i] * temp

    return return_value


def log_range(x_array, y_array):
    print("метод мистера Лагранжа")

    build_points(x_array, y_array)

    points = np.linspace(x_array[0], x_array[len(x_array) - 1], 228)
    plt.plot(points, knowledge_of_maya(x_array, y_array, points))

    plt.grid(True)
    plt.axis([START_X, END_X, START_Y, END_Y])
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.show()


def letting_kraken_out(point, start, end):
    global x
    global y

    if abs(start - end) == 1:
        matrix = np.array([[point - x[start], y[start]],
                           [point - x[end], y[end]]])
    else:
        matrix = np.array([[point - x[start], letting_kraken_out(point, start, end - 1)],
                           [point - x[end], letting_kraken_out(point, start + 1, end)]])
    return 1 / (x[end] - x[start]) * np.linalg.det(matrix)


def hay_taken(point):
    global x
    print("метод дяди Эйткена")
    print('P({0}) = {1}'.format(point, letting_kraken_out(point, 0, len(x) - 1)))


if __name__ == "__main__":

    try:
        log_range(x, y)
    except Exception as e:
        print(e)

    try:
        hay_taken(point)
    except Exception as e:
        print(e)
