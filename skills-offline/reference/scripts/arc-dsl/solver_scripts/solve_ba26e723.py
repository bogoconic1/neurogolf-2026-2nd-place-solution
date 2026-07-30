from dsl import *
from constants import *


def solve_ba26e723(I):
    x1 = rbind(divide, THREE)
    x2 = rbind(multiply, THREE)
    x3 = compose(x2, x1)
    x4 = fork(equality, identity, x3)
    x5 = compose(x4, last)
    x6 = ofcolor(I, FOUR)
    x7 = sfilter(x6, x5)
    O = fill(I, SIX, x7)
    return O
