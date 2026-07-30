from dsl import *
from constants import *


def solve_ef135b50(I):
    x1 = ofcolor(I, TWO)
    x2 = ofcolor(I, ZERO)
    x3 = product(x1, x1)
    x4 = power(first, TWO)
    x5 = compose(first, last)
    x6 = fork(equality, x4, x5)
    x7 = sfilter(x3, x6)
    x8 = fork(connect, first, last)
    x9 = mapply(x8, x7)
    x10 = intersection(x9, x2)
    x11 = fill(I, NINE, x10)
    x12 = trim(x11)
    x13 = asobject(x12)
    x14 = shift(x13, UNITY)
    O = paint(I, x14)
    return O
