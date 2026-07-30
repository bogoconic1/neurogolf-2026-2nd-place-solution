from dsl import *
from constants import *


def solve_d90796e8(I):
    x1 = objects(I, F, F, T)
    x2 = sizefilter(x1, TWO)
    x3 = lbind(contained, TWO)
    x4 = compose(x3, palette)
    x5 = mfilter(x2, x4)
    x6 = cover(I, x5)
    x7 = matcher(first, THREE)
    x8 = sfilter(x5, x7)
    O = fill(x6, EIGHT, x8)
    return O
