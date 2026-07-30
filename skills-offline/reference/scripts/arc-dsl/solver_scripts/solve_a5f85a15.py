from dsl import *
from constants import *


def solve_a5f85a15(I):
    x1 = objects(I, T, T, T)
    x2 = interval(ONE, NINE, ONE)
    x3 = apply(double, x2)
    x4 = apply(decrement, x3)
    x5 = papply(astuple, x4, x4)
    x6 = apply(ulcorner, x1)
    x7 = lbind(shift, x5)
    x8 = mapply(x7, x6)
    O = fill(I, FOUR, x8)
    return O
