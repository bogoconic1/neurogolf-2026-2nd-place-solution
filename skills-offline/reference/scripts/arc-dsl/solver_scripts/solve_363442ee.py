from dsl import *
from constants import *


def solve_363442ee(I):
    x1 = ofcolor(I, ONE)
    x2 = crop(I, ORIGIN, THREE_BY_THREE)
    x3 = asobject(x2)
    x4 = lbind(shift, x3)
    x5 = compose(x4, decrement)
    x6 = mapply(x5, x1)
    O = paint(I, x6)
    return O
