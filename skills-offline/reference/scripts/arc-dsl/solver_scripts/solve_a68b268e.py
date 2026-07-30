from dsl import *
from constants import *


def solve_a68b268e(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = lefthalf(x1)
    x4 = righthalf(x1)
    x5 = lefthalf(x2)
    x6 = righthalf(x2)
    x7 = ofcolor(x4, FOUR)
    x8 = ofcolor(x3, SEVEN)
    x9 = ofcolor(x5, EIGHT)
    x10 = fill(x6, EIGHT, x9)
    x11 = fill(x10, FOUR, x7)
    O = fill(x11, SEVEN, x8)
    return O
