from dsl import *
from constants import *


def solve_ec883f72(I):
    x1 = palette(I)
    x2 = objects(I, T, T, T)
    x3 = fork(multiply, height, width)
    x4 = argmax(x2, x3)
    x5 = color(x4)
    x6 = remove(ZERO, x1)
    x7 = other(x6, x5)
    x8 = lrcorner(x4)
    x9 = llcorner(x4)
    x10 = urcorner(x4)
    x11 = ulcorner(x4)
    x12 = shoot(x8, UNITY)
    x13 = shoot(x9, DOWN_LEFT)
    x14 = shoot(x10, UP_RIGHT)
    x15 = shoot(x11, NEG_UNITY)
    x16 = combine(x12, x13)
    x17 = combine(x14, x15)
    x18 = combine(x16, x17)
    O = underfill(I, x7, x18)
    return O
