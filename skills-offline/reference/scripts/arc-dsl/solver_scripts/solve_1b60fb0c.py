from dsl import *
from constants import *


def solve_1b60fb0c(I):
    x1 = rot90(I)
    x2 = ofcolor(I, ONE)
    x3 = ofcolor(x1, ONE)
    x4 = neighbors(ORIGIN)
    x5 = mapply(neighbors, x4)
    x6 = lbind(shift, x3)
    x7 = apply(x6, x5)
    x8 = lbind(intersection, x2)
    x9 = compose(size, x8)
    x10 = argmax(x7, x9)
    O = underfill(I, TWO, x10)
    return O
