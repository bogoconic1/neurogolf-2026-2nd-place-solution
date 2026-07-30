from dsl import *
from constants import *


def solve_94f9d214(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = ofcolor(x1, ZERO)
    x4 = ofcolor(x2, ZERO)
    x5 = astuple(FOUR, FOUR)
    x6 = canvas(ZERO, x5)
    x7 = intersection(x3, x4)
    O = fill(x6, TWO, x7)
    return O
