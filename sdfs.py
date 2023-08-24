import numpy as np


def one_point(x):
    x -= x.mean()
    return x.mean() - x


def one_wall(x):
    x -= x.mean()
    return np.abs(x) - 0.5


def two_walls(x):
    x -= x.mean()
    return np.minimum(np.abs(x - 0.5) - 0.25, np.abs(x + 0.5) - 0.25)


sdfs = {"one_point": one_point, "one_wall": one_wall, "two_walls": two_walls}
