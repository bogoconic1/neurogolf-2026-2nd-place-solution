from dsl import *
from constants import *


def solve_3bd67248(I):
    x1 = height(I)
    x2 = decrement(x1)
    x3 = decrement(x2)
    x4 = astuple(x3, ONE)
    x5 = astuple(x2, ONE)
    x6 = shoot(x4, UP_RIGHT)
    x7 = shoot(x5, RIGHT)
    x8 = fill(I, TWO, x6)
    O = fill(x8, FOUR, x7)
    return O
