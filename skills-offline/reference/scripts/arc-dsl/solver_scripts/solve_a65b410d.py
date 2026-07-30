from dsl import *
from constants import *


def solve_a65b410d(I):
    x1 = ofcolor(I, TWO)
    x2 = urcorner(x1)
    x3 = shoot(x2, UP_RIGHT)
    x4 = shoot(x2, DOWN_LEFT)
    x5 = underfill(I, THREE, x3)
    x6 = underfill(x5, ONE, x4)
    x7 = rbind(shoot, LEFT)
    x8 = mapply(x7, x3)
    x9 = mapply(x7, x4)
    x10 = underfill(x6, ONE, x9)
    O = underfill(x10, THREE, x8)
    return O
