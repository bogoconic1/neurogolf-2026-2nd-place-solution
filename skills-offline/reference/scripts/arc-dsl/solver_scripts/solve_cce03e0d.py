from dsl import *
from constants import *


def solve_cce03e0d(I):
    x1 = upscale(I, THREE)
    x2 = hconcat(I, I)
    x3 = hconcat(x2, I)
    x4 = vconcat(x3, x3)
    x5 = vconcat(x4, x3)
    x6 = ofcolor(x1, ZERO)
    x7 = ofcolor(x1, ONE)
    x8 = combine(x6, x7)
    O = fill(x5, ZERO, x8)
    return O
