from dsl import *
from constants import *


def solve_e9614598(I):
    x1 = ofcolor(I, ONE)
    x2 = fork(add, first, last)
    x3 = x2(x1)
    x4 = halve(x3)
    x5 = dneighbors(x4)
    x6 = insert(x4, x5)
    O = fill(I, THREE, x6)
    return O
