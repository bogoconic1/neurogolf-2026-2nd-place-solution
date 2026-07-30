from dsl import *
from constants import *


def solve_77fdfe62(I):
    x1 = ofcolor(I, EIGHT)
    x2 = subgrid(x1, I)
    x3 = replace(I, EIGHT, ZERO)
    x4 = replace(x3, ONE, ZERO)
    x5 = compress(x4)
    x6 = width(x2)
    x7 = halve(x6)
    x8 = upscale(x5, x7)
    x9 = ofcolor(x2, ZERO)
    O = fill(x8, ZERO, x9)
    return O
