from dsl import *
from constants import *


def solve_fafffa47(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = shape(x2)
    x4 = ofcolor(x1, ZERO)
    x5 = ofcolor(x2, ZERO)
    x6 = intersection(x4, x5)
    x7 = canvas(ZERO, x3)
    O = fill(x7, TWO, x6)
    return O
