from dsl import *
from constants import *


def solve_a87f7484(I):
    x1 = numcolors(I)
    x2 = dmirror(I)
    x3 = portrait(I)
    x4 = branch(x3, dmirror, identity)
    x5 = x4(I)
    x6 = decrement(x1)
    x7 = hsplit(x5, x6)
    x8 = rbind(ofcolor, ZERO)
    x9 = apply(x8, x7)
    x10 = leastcommon(x9)
    x11 = matcher(x8, x10)
    x12 = extract(x7, x11)
    O = x4(x12)
    return O
