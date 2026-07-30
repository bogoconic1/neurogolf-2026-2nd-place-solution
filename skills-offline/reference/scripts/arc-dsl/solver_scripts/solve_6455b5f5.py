from dsl import *
from constants import *


def solve_6455b5f5(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = argmax(x1, size)
    x4 = valmin(x1, size)
    x5 = sizefilter(x2, x4)
    x6 = recolor(ONE, x3)
    x7 = merge(x5)
    x8 = paint(I, x6)
    O = fill(x8, EIGHT, x7)
    return O
