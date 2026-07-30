from dsl import *
from constants import *


def solve_83302e8f(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = sfilter(x2, square)
    x4 = difference(x2, x3)
    x5 = merge(x3)
    x6 = recolor(THREE, x5)
    x7 = merge(x4)
    x8 = recolor(FOUR, x7)
    x9 = paint(I, x6)
    O = paint(x9, x8)
    return O
