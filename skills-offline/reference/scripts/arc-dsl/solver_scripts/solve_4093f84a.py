from dsl import *
from constants import *


def solve_4093f84a(I):
    x1 = leastcolor(I)
    x2 = replace(I, x1, FIVE)
    x3 = ofcolor(I, FIVE)
    x4 = portrait(x3)
    x5 = branch(x4, identity, dmirror)
    x6 = x5(x2)
    x7 = lefthalf(x6)
    x8 = righthalf(x6)
    x9 = rbind(order, identity)
    x10 = rbind(order, invert)
    x11 = apply(x9, x7)
    x12 = apply(x10, x8)
    x13 = hconcat(x11, x12)
    O = x5(x13)
    return O
