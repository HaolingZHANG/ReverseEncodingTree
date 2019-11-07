import math
import numpy


def rastrigin(x, y, a=10):
    return a + sum([(index**2 - a * numpy.cos(2 * math.pi * index)) for index in (x, y)])


def save_terrain(matrix, path="../dataset/rastrigin.csv"):
    with open(path, "w", encoding="utf-8") as save_file:
        for row_data in matrix:
            string = str(row_data)[1:-1]
            string = string.replace("\'", "")
            string = string.replace(" ", "")
            save_file.write(string + "\n")


if __name__ == '__main__':
    X = numpy.linspace(-4, 4, 200)
    Y = numpy.linspace(-4, 4, 200)

    X, Y = numpy.meshgrid(X, Y)

    Z = rastrigin(X, Y)

    outputs = []
    for data in Z:
        outputs.append(list(data))

    save_terrain(outputs)

    print(numpy.max(Z))
    print(numpy.min(Z))
