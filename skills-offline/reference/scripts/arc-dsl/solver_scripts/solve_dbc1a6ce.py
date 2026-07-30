from dsl import *
from constants import *


def solve_dbc1a6ce(I):
    x1 = ofcolor(I, ONE)
    x2 = product(x1, x1)
    x3 = fork(connect, first, last)
    x4 = apply(x3, x2)
    x5 = fork(either, vline, hline)
    x6 = mfilter(x4, x5)
    O = underfill(I, EIGHT, x6)
    return O
