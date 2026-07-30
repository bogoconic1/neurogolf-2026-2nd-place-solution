from dsl import *
from constants import *


def solve_ff28f65a(I):
    x1 = objects(I, T, F, T)
    x2 = size(x1)
    x3 = double(x2)
    x4 = interval(ZERO, x3, TWO)
    x5 = apply(tojvec, x4)
    x6 = astuple(ONE, NINE)
    x7 = canvas(ZERO, x6)
    x8 = fill(x7, ONE, x5)
    x9 = hsplit(x8, THREE)
    O = merge(x9)
    return O
