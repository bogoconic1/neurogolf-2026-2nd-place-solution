from dsl import *
from constants import *


def solve_a3df8b1e(I):
    x1 = shape(I)
    x2 = ofcolor(I, ONE)
    x3 = first(x2)
    x4 = shoot(x3, UP_RIGHT)
    x5 = fill(I, ONE, x4)
    x6 = ofcolor(x5, ONE)
    x7 = urcorner(x6)
    x8 = shoot(x7, NEG_UNITY)
    x9 = fill(x5, ONE, x8)
    x10 = objects(x9, T, T, T)
    x11 = first(x10)
    x12 = subgrid(x11, x9)
    x13 = shape(x12)
    x14 = subtract(x13, DOWN)
    x15 = crop(x12, DOWN, x14)
    x16 = vconcat(x15, x15)
    x17 = vconcat(x16, x16)
    x18 = vconcat(x17, x17)
    x19 = hmirror(x18)
    x20 = crop(x19, ORIGIN, x1)
    O = hmirror(x20)
    return O
