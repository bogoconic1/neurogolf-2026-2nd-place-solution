from dsl import *
from constants import *


def solve_d2abd087(I):
    x1 = objects(I, T, F, T)
    x2 = matcher(size, SIX)
    x3 = compose(flip, x2)
    x4 = mfilter(x1, x2)
    x5 = mfilter(x1, x3)
    x6 = fill(I, TWO, x4)
    O = fill(x6, ONE, x5)
    return O
