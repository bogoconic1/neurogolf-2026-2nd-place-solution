from dsl import *
from constants import *


def solve_8eb1be9a(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = interval(NEG_TWO, FOUR, ONE)
    x4 = lbind(shift, x2)
    x5 = height(x2)
    x6 = rbind(multiply, x5)
    x7 = apply(x6, x3)
    x8 = apply(toivec, x7)
    x9 = mapply(x4, x8)
    O = paint(I, x9)
    return O
