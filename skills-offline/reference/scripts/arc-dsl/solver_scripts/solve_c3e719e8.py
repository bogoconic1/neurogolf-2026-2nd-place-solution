from dsl import *
from constants import *


def solve_c3e719e8(I):
    x1 = mostcolor(I)
    x2 = hconcat(I, I)
    x3 = upscale(I, THREE)
    x4 = ofcolor(x3, x1)
    x5 = asindices(x3)
    x6 = difference(x5, x4)
    x7 = hconcat(x2, I)
    x8 = vconcat(x7, x7)
    x9 = vconcat(x8, x7)
    O = fill(x9, ZERO, x6)
    return O
