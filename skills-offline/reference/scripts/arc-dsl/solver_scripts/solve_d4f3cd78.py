from dsl import *
from constants import *


def solve_d4f3cd78(I):
    x1 = ofcolor(I, FIVE)
    x2 = delta(x1)
    x3 = fill(I, EIGHT, x2)
    x4 = box(x1)
    x5 = difference(x4, x1)
    x6 = position(x4, x5)
    x7 = first(x5)
    x8 = shoot(x7, x6)
    O = fill(x3, EIGHT, x8)
    return O
