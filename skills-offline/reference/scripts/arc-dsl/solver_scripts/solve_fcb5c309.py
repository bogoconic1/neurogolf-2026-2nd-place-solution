from dsl import *
from constants import *


def solve_fcb5c309(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = colorfilter(x2, x1)
    x4 = difference(x2, x3)
    x5 = argmax(x4, size)
    x6 = color(x5)
    x7 = subgrid(x5, I)
    O = replace(x7, x6, x1)
    return O
