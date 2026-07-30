from dsl import *
from constants import *


def solve_bbc9ae5d(I):
    x1 = width(I)
    x2 = palette(I)
    x3 = halve(x1)
    x4 = vupscale(I, x3)
    x5 = rbind(shoot, UNITY)
    x6 = other(x2, ZERO)
    x7 = ofcolor(x4, x6)
    x8 = mapply(x5, x7)
    O = fill(x4, x6, x8)
    return O
