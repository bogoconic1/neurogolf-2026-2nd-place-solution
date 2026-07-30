from dsl import *
from constants import *


def solve_dc0a314f(I):
    x1 = ofcolor(I, THREE)
    x2 = replace(I, THREE, ZERO)
    x3 = dmirror(x2)
    x4 = papply(pair, x2, x3)
    x5 = lbind(apply, maximum)
    x6 = apply(x5, x4)
    x7 = cmirror(x6)
    x8 = papply(pair, x6, x7)
    x9 = apply(x5, x8)
    O = subgrid(x1, x9)
    return O
