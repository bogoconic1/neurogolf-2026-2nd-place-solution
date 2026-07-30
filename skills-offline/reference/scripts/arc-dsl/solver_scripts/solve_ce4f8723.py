from dsl import *
from constants import *


def solve_ce4f8723(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = ofcolor(x1, ZERO)
    x4 = ofcolor(x2, ZERO)
    x5 = intersection(x3, x4)
    x6 = astuple(FOUR, FOUR)
    x7 = canvas(THREE, x6)
    O = fill(x7, ZERO, x5)
    return O
