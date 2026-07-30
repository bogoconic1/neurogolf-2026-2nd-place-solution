from dsl import *
from constants import *


def solve_ff805c23(I):
    x1 = hmirror(I)
    x2 = vmirror(I)
    x3 = ofcolor(I, ONE)
    x4 = subgrid(x3, x1)
    x5 = subgrid(x3, x2)
    x6 = palette(x4)
    x7 = contained(ONE, x6)
    O = branch(x7, x5, x4)
    return O
