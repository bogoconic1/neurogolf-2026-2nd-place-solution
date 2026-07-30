from dsl import *
from constants import *


def solve_dae9d2b5(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = ofcolor(x1, FOUR)
    x4 = ofcolor(x2, THREE)
    x5 = combine(x3, x4)
    O = fill(x1, SIX, x5)
    return O
