from dsl import *
from constants import *


def solve_b60334d2(I):
    x1 = ofcolor(I, FIVE)
    x2 = replace(I, FIVE, ZERO)
    x3 = mapply(dneighbors, x1)
    x4 = mapply(ineighbors, x1)
    x5 = fill(x2, ONE, x3)
    O = fill(x5, FIVE, x4)
    return O
