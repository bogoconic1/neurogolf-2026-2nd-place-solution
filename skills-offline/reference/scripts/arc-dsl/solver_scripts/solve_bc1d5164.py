from dsl import *
from constants import *


def solve_bc1d5164(I):
    x1 = leastcolor(I)
    x2 = crop(I, ORIGIN, THREE_BY_THREE)
    x3 = crop(I, TWO_BY_ZERO, THREE_BY_THREE)
    x4 = tojvec(FOUR)
    x5 = crop(I, x4, THREE_BY_THREE)
    x6 = astuple(TWO, FOUR)
    x7 = crop(I, x6, THREE_BY_THREE)
    x8 = canvas(ZERO, THREE_BY_THREE)
    x9 = rbind(ofcolor, x1)
    x10 = astuple(x2, x3)
    x11 = astuple(x5, x7)
    x12 = combine(x10, x11)
    x13 = mapply(x9, x12)
    O = fill(x8, x1, x13)
    return O
