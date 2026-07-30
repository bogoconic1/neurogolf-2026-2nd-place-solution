from dsl import *
from constants import *


def solve_af902bf9(I):
    x1 = ofcolor(I, FOUR)
    x2 = prapply(connect, x1, x1)
    x3 = fork(either, vline, hline)
    x4 = mfilter(x2, x3)
    x5 = underfill(I, NEG_ONE, x4)
    x6 = objects(x5, F, F, T)
    x7 = compose(backdrop, inbox)
    x8 = mapply(x7, x6)
    x9 = fill(x5, TWO, x8)
    O = replace(x9, NEG_ONE, ZERO)
    return O
