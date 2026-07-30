from dsl import *
from constants import *


def solve_95990924(I):
    x1 = objects(I, T, F, T)
    x2 = apply(outbox, x1)
    x3 = apply(ulcorner, x2)
    x4 = apply(urcorner, x2)
    x5 = apply(llcorner, x2)
    x6 = apply(lrcorner, x2)
    x7 = fill(I, ONE, x3)
    x8 = fill(x7, TWO, x4)
    x9 = fill(x8, THREE, x5)
    O = fill(x9, FOUR, x6)
    return O
