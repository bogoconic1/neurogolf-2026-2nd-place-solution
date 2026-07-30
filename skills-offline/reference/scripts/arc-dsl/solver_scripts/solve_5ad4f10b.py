from dsl import *
from constants import *


def solve_5ad4f10b(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    x3 = color(x2)
    x4 = subgrid(x2, I)
    x5 = leastcolor(x4)
    x6 = replace(x4, x5, ZERO)
    x7 = replace(x6, x3, x5)
    x8 = height(x7)
    x9 = divide(x8, THREE)
    O = downscale(x7, x9)
    return O
