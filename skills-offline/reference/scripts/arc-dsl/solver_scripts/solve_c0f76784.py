from dsl import *
from constants import *


def solve_c0f76784(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = sfilter(x2, square)
    x4 = sizefilter(x3, ONE)
    x5 = merge(x4)
    x6 = argmax(x3, size)
    x7 = merge(x3)
    x8 = fill(I, SEVEN, x7)
    x9 = fill(x8, EIGHT, x6)
    O = fill(x9, SIX, x5)
    return O
