from dsl import *
from constants import *


def solve_7c008303(I):
    x1 = ofcolor(I, THREE)
    x2 = subgrid(x1, I)
    x3 = ofcolor(x2, ZERO)
    x4 = replace(I, THREE, ZERO)
    x5 = replace(x4, EIGHT, ZERO)
    x6 = compress(x5)
    x7 = upscale(x6, THREE)
    O = fill(x7, ZERO, x3)
    return O
