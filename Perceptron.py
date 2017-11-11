import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import datasets


def conv(s):
    try:
        s = float(s)
    except ValueError:
        pass
    return s


def tipo(s):
    return {
        "setosa": [0, 0],
        "versicolor": [0, 1],
        "virginica": [1, 0]
    }.get(s, [1, 1])


def read_data(file):
    with open(file) as f:
        reader = csv.reader(f)
        data = []
        dataOut = []
        for row in reader:
            data.append([conv(x) for x in row][0:1])
            dataOut.append(tipo(row[8]))

        return data, dataOut


class Perceptron(object):

    itr = 10000

    def __init__(self, x, y):
        self.nIn = len(x[0])
        self.nOut = len(y[0])
        self.X = x
        self.Y = y

        self.nTrain = 1

        self.B = []
        self.W = []

        for i in range(self.nOut):
            self.B.append(1)
            m = []
            for j in range(self.nIn):
                m.append(1)
            self.W.append(m)

    def multiM(self, a, b):
        r = []
        r.append([0]*len(b[0]))

    def activation(self, x, id_s):
        r = np.dot(x, self.W[id_s])
        if r - self.B[id_s] >= 0:
            return 1
        else:
            return 0

    def train(self):
        error = []
        for i in range(self.itr):
            er = 0
            for j in range(len(self.X)):
                for k in range(self.nOut):
                    yi = self.activation(self.X[j], k)
                    ye = self.Y[j][k]

                    e = ye - yi
                    er += abs(e)

                    if e != 0:
                        for l in range(self.nIn):
                            self.W[k][l] += self.nTrain * self.X[j][l] * e

            error.append(er)
            if er == 0:
                break

        plt.plot(error)
        plt.show()






X = [
    [1, 2],
    [-1, 1],
    [0, -1]
]

Y = [
    [1, 1],
    [0, 1],
    [1, 0]
]
rx, ry = read_data('Dataset/IRIS.csv')

iris = datasets.load_iris()
dx = iris.data[:,:2]
dy = iris.target
print(dx)
P = Perceptron(rx[0:10], ry[0:10])

P.train()

