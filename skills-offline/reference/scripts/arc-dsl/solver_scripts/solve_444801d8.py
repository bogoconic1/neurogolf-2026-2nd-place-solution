from dsl import *
from constants import *


def solve_444801d8(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, ONE)
    x3 = rbind(toobject, I)
    x4 = chain(leastcolor, x3, delta)
    x5 = rbind(shift, UP)
    x6 = compose(x5, backdrop)
    x7 = fork(recolor, x4, x6)
    x8 = mapply(x7, x2)
    O = underpaint(I, x8)
    return O
