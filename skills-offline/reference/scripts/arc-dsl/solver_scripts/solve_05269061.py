from dsl import *
from constants import *


def solve_05269061(I):
    x1 = objects(I, T, T, T)
    x2 = neighbors(ORIGIN)
    x3 = mapply(neighbors, x2)
    x4 = rbind(multiply, THREE)
    x5 = apply(x4, x3)
    x6 = merge(x1)
    x7 = lbind(shift, x6)
    x8 = mapply(x7, x5)
    x9 = shift(x8, UP_RIGHT)
    x10 = shift(x8, DOWN_LEFT)
    x11 = paint(I, x8)
    x12 = paint(x11, x9)
    O = paint(x12, x10)
    return O
