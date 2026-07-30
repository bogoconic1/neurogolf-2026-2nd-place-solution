from dsl import *
from constants import *


def solve_36fdfd69(I):
    x1 = upscale(I, TWO)
    x2 = objects(x1, T, T, T)
    x3 = colorfilter(x2, TWO)
    x4 = fork(manhattan, first, last)
    x5 = lbind(greater, FIVE)
    x6 = compose(x5, x4)
    x7 = product(x3, x3)
    x8 = sfilter(x7, x6)
    x9 = apply(merge, x8)
    x10 = mapply(delta, x9)
    x11 = fill(x1, FOUR, x10)
    x12 = merge(x3)
    x13 = paint(x11, x12)
    O = downscale(x13, TWO)
    return O
