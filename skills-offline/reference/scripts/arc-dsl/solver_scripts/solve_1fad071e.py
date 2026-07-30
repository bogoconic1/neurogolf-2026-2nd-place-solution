from dsl import *
from constants import *


def solve_1fad071e(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, ONE)
    x3 = sizefilter(x2, FOUR)
    x4 = size(x3)
    x5 = subtract(FIVE, x4)
    x6 = astuple(ONE, x4)
    x7 = canvas(ONE, x6)
    x8 = astuple(ONE, x5)
    x9 = canvas(ZERO, x8)
    O = hconcat(x7, x9)
    return O
