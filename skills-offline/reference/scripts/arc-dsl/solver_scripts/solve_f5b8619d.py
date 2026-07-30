from dsl import *
from constants import *


def solve_f5b8619d(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = mapply(vfrontier, x2)
    x4 = underfill(I, EIGHT, x3)
    x5 = hconcat(x4, x4)
    O = vconcat(x5, x5)
    return O
