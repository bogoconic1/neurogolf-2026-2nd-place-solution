from dsl import *
from constants import *


def solve_c9f8e694(I):
    x1 = height(I)
    x2 = width(I)
    x3 = ofcolor(I, ZERO)
    x4 = astuple(x1, ONE)
    x5 = crop(I, ORIGIN, x4)
    x6 = hupscale(x5, x2)
    O = fill(x6, ZERO, x3)
    return O
