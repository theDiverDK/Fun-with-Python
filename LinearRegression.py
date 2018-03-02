from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)
#ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float)


def createDataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val+random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float), np.array(ys, dtype=np.float)


def bestFitSlopeAndIntercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) ** 2) - mean(xs ** 2)))

    b = mean(ys) - m * mean(xs)
    return m, b


def squaredError(ysOriginal, ysLine):
    return sum((ysLine - ysOriginal) ** 2)


def coefficienOfDetermination(ysOriginal, ysLine):
    yMeanLine = [mean(ysOriginal) for y in ysOriginal]
    squaredErrorRegression = squaredError(ysOriginal, ysLine)
    squaredErrorYMean = squaredError(ysOriginal, yMeanLine)
    return 1 - (squaredErrorRegression / squaredErrorYMean)


xs, ys = createDataset(40, 40, 2, correlation='pos')

m, b = bestFitSlopeAndIntercept(xs, ys)

print(m, b)

regressionLine = [(m * x + b) for x in xs]

r2 = coefficienOfDetermination(ys, regressionLine)

print(r2)

plt.scatter(xs, ys)
plt.plot(xs, regressionLine)
plt.show()

# plt.scatter(xs,ys)
# plt.show()
