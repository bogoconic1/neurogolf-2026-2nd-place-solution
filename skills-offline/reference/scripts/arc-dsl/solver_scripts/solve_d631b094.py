from dsl import *
from constants import *


def solve_d631b094(I):
    x1 = palette(I)
    x2 = other(x1, ZERO)
    x3 = ofcolor(I, x2)
    x4 = size(x3)
    x5 = astuple(ONE, x4)
    O = canvas(x2, x5)
    return O
