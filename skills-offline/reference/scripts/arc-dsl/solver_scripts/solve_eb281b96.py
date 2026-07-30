from dsl import *
from constants import *


def solve_eb281b96(I):
    x1 = height(I)
    x2 = width(I)
    x3 = decrement(x1)
    x4 = astuple(x3, x2)
    x5 = crop(I, ORIGIN, x4)
    x6 = hmirror(x5)
    x7 = vconcat(I, x6)
    x8 = double(x3)
    x9 = astuple(x8, x2)
    x10 = crop(x7, DOWN, x9)
    O = vconcat(x7, x10)
    return O
