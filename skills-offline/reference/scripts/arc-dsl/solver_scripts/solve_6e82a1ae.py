from dsl import *
from constants import *


def solve_6e82a1ae(I):
    x1 = objects(I, T, F, T)
    x2 = lbind(sizefilter, x1)
    x3 = compose(merge, x2)
    x4 = x3(TWO)
    x5 = x3(THREE)
    x6 = x3(FOUR)
    x7 = fill(I, THREE, x4)
    x8 = fill(x7, TWO, x5)
    O = fill(x8, ONE, x6)
    return O
