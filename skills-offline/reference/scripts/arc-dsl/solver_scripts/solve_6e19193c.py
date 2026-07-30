from dsl import *
from constants import *


def solve_6e19193c(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = rbind(toobject, I)
    x4 = compose(first, delta)
    x5 = rbind(colorcount, x1)
    x6 = matcher(x5, TWO)
    x7 = chain(x6, x3, dneighbors)
    x8 = rbind(sfilter, x7)
    x9 = chain(first, x8, toindices)
    x10 = fork(subtract, x4, x9)
    x11 = fork(shoot, x4, x10)
    x12 = mapply(x11, x2)
    x13 = fill(I, x1, x12)
    x14 = mapply(delta, x2)
    O = fill(x13, ZERO, x14)
    return O
