from dsl import *
from constants import *


def solve_7447852a(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = compose(last, center)
    x4 = order(x2, x3)
    x5 = size(x4)
    x6 = interval(ZERO, x5, THREE)
    x7 = rbind(contained, x6)
    x8 = compose(x7, last)
    x9 = interval(ZERO, x5, ONE)
    x10 = pair(x4, x9)
    x11 = sfilter(x10, x8)
    x12 = mapply(first, x11)
    O = fill(I, FOUR, x12)
    return O
