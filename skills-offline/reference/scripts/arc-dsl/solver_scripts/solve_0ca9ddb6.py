from dsl import *
from constants import *


def solve_0ca9ddb6(I):
    x1 = ofcolor(I, ONE)
    x2 = ofcolor(I, TWO)
    x3 = mapply(dneighbors, x1)
    x4 = mapply(ineighbors, x2)
    x5 = fill(I, SEVEN, x3)
    O = fill(x5, FOUR, x4)
    return O
