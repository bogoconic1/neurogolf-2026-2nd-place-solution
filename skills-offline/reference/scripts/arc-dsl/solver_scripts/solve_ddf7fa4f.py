from dsl import *
from constants import *


def solve_ddf7fa4f(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = colorfilter(x1, FIVE)
    x4 = product(x2, x3)
    x5 = fork(vmatching, first, last)
    x6 = sfilter(x4, x5)
    x7 = compose(color, first)
    x8 = fork(recolor, x7, last)
    x9 = mapply(x8, x6)
    O = paint(I, x9)
    return O
