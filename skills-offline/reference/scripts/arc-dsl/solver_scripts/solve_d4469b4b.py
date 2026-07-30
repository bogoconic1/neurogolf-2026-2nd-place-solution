from dsl import *
from constants import *


def solve_d4469b4b(I):
    x1 = palette(I)
    x2 = other(x1, ZERO)
    x3 = equality(x2, ONE)
    x4 = equality(x2, TWO)
    x5 = branch(x3, UNITY, TWO_BY_TWO)
    x6 = branch(x4, RIGHT, x5)
    x7 = fork(combine, vfrontier, hfrontier)
    x8 = x7(x6)
    x9 = canvas(ZERO, THREE_BY_THREE)
    O = fill(x9, FIVE, x8)
    return O
