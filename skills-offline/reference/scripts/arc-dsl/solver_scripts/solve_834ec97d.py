from dsl import *
from constants import *


def solve_834ec97d(I):
    x1 = asindices(I)
    x2 = objects(I, T, F, T)
    x3 = first(x2)
    x4 = shift(x3, DOWN)
    x5 = fill(I, ZERO, x3)
    x6 = paint(x5, x4)
    x7 = uppermost(x4)
    x8 = leftmost(x4)
    x9 = subtract(x8, TEN)
    x10 = add(x8, TEN)
    x11 = interval(x9, x10, TWO)
    x12 = lbind(greater, x7)
    x13 = compose(x12, first)
    x14 = rbind(contained, x11)
    x15 = compose(x14, last)
    x16 = sfilter(x1, x13)
    x17 = sfilter(x16, x15)
    O = fill(x6, FOUR, x17)
    return O
