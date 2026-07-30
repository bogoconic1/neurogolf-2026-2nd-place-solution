from dsl import *
from constants import *


def solve_e9afcf9a(I):
    x1 = astuple(TWO, ONE)
    x2 = crop(I, ORIGIN, x1)
    x3 = hmirror(x2)
    x4 = hconcat(x2, x3)
    x5 = hconcat(x4, x4)
    O = hconcat(x5, x4)
    return O
