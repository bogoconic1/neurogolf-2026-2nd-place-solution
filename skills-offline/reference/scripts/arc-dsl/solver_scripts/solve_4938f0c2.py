from dsl import *
from constants import *


def solve_4938f0c2(I):
    x1 = objects(I, T, T, T)
    x2 = ofcolor(I, TWO)
    x3 = vmirror(x2)
    x4 = height(x2)
    x5 = width(x2)
    x6 = toivec(x4)
    x7 = tojvec(x5)
    x8 = add(x7, ZERO_BY_TWO)
    x9 = add(x6, TWO_BY_ZERO)
    x10 = shift(x3, x8)
    x11 = fill(I, TWO, x10)
    x12 = ofcolor(x11, TWO)
    x13 = hmirror(x12)
    x14 = shift(x13, x9)
    x15 = fill(x11, TWO, x14)
    x16 = size(x1)
    x17 = greater(x16, FOUR)
    O = branch(x17, I, x15)
    return O
