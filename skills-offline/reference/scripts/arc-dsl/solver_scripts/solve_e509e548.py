from dsl import *
from constants import *


def solve_e509e548(I):
    x1 = objects(I, T, F, T)
    x2 = rbind(subgrid, I)
    x3 = chain(palette, trim, x2)
    x4 = lbind(contained, THREE)
    x5 = compose(x4, x3)
    x6 = fork(add, height, width)
    x7 = compose(decrement, x6)
    x8 = fork(equality, size, x7)
    x9 = mfilter(x1, x5)
    x10 = mfilter(x1, x8)
    x11 = replace(I, THREE, SIX)
    x12 = fill(x11, TWO, x9)
    O = fill(x12, ONE, x10)
    return O
