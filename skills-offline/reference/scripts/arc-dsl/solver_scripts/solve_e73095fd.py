from dsl import *
from constants import *


def solve_e73095fd(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = fork(equality, toindices, backdrop)
    x4 = sfilter(x2, x3)
    x5 = lbind(mapply, dneighbors)
    x6 = chain(x5, corners, outbox)
    x7 = fork(difference, x6, outbox)
    x8 = ofcolor(I, FIVE)
    x9 = rbind(intersection, x8)
    x10 = matcher(size, ZERO)
    x11 = chain(x10, x9, x7)
    x12 = mfilter(x4, x11)
    O = fill(I, FOUR, x12)
    return O
