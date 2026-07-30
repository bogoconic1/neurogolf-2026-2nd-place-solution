from dsl import *
from constants import *


def solve_db93a21d(I):
    x1 = objects(I, T, T, T)
    x2 = ofcolor(I, NINE)
    x3 = colorfilter(x1, NINE)
    x4 = rbind(shoot, DOWN)
    x5 = mapply(x4, x2)
    x6 = underfill(I, ONE, x5)
    x7 = compose(halve, width)
    x8 = rbind(greater, ONE)
    x9 = compose(x8, x7)
    x10 = matcher(x7, THREE)
    x11 = power(outbox, TWO)
    x12 = power(outbox, THREE)
    x13 = mapply(outbox, x3)
    x14 = sfilter(x3, x9)
    x15 = sfilter(x3, x10)
    x16 = mapply(x11, x14)
    x17 = mapply(x12, x15)
    x18 = fill(x6, THREE, x13)
    x19 = fill(x18, THREE, x16)
    O = fill(x19, THREE, x17)
    return O
