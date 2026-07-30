from dsl import *
from constants import *


def solve_29623171(I):
    x1 = leastcolor(I)
    x2 = interval(ZERO, NINE, FOUR)
    x3 = product(x2, x2)
    x4 = rbind(add, THREE)
    x5 = rbind(interval, ONE)
    x6 = fork(x5, identity, x4)
    x7 = compose(x6, first)
    x8 = compose(x6, last)
    x9 = fork(product, x7, x8)
    x10 = rbind(colorcount, x1)
    x11 = rbind(toobject, I)
    x12 = compose(x10, x11)
    x13 = apply(x9, x3)
    x14 = valmax(x13, x12)
    x15 = matcher(x12, x14)
    x16 = compose(flip, x15)
    x17 = mfilter(x13, x15)
    x18 = mfilter(x13, x16)
    x19 = fill(I, x1, x17)
    O = fill(x19, ZERO, x18)
    return O
