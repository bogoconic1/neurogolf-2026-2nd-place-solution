from dsl import *
from constants import *


def solve_855e0971(I):
    x1 = rot90(I)
    x2 = frontiers(I)
    x3 = sfilter(x2, hline)
    x4 = size(x3)
    x6 = positive(x4)
    x7 = branch(x6, identity, dmirror)
    x8 = x7(I)
    x9 = rbind(subgrid, x8)
    x10 = matcher(color, ZERO)
    x11 = compose(flip, x10)
    x12 = partition(x8)
    x13 = sfilter(x12, x11)
    x14 = rbind(ofcolor, ZERO)
    x15 = lbind(mapply, vfrontier)
    x16 = chain(x15, x14, x9)
    x17 = fork(shift, x16, ulcorner)
    x18 = fork(intersection, toindices, x17)
    x19 = mapply(x18, x13)
    x20 = fill(x8, ZERO, x19)
    O = x7(x20)
    return O
