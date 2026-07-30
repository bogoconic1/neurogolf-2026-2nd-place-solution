from dsl import *
from constants import *


def solve_1f642eb9(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = difference(x1, x2)
    x4 = first(x3)
    x5 = rbind(gravitate, x4)
    x6 = compose(crement, x5)
    x7 = fork(shift, identity, x6)
    x8 = mapply(x7, x2)
    O = paint(I, x8)
    return O
