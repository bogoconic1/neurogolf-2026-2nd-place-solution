from dsl import *
from constants import *


def solve_05f2a901(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, TWO)
    x3 = first(x2)
    x4 = colorfilter(x1, EIGHT)
    x5 = first(x4)
    x6 = gravitate(x3, x5)
    O = move(I, x3, x6)
    return O
