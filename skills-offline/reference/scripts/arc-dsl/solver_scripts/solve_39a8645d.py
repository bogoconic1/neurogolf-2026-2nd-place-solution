from dsl import *
from constants import *


def solve_39a8645d(I):
    x1 = objects(I, T, T, T)
    x2 = totuple(x1)
    x3 = apply(color, x2)
    x4 = mostcommon(x3)
    x5 = matcher(color, x4)
    x6 = extract(x1, x5)
    O = subgrid(x6, I)
    return O
