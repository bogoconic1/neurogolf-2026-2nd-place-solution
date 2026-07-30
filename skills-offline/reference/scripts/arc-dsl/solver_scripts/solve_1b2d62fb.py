from dsl import *
from constants import *


def solve_1b2d62fb(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = ofcolor(x1, ZERO)
    x4 = ofcolor(x2, ZERO)
    x5 = intersection(x3, x4)
    x6 = replace(x1, NINE, ZERO)
    O = fill(x6, EIGHT, x5)
    return O
