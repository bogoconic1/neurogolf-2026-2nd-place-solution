from dsl import *
from constants import *


def solve_9af7a82c(I):
    x1 = objects(I, T, F, F)
    x2 = order(x1, size)
    x3 = valmax(x1, size)
    x4 = rbind(astuple, ONE)
    x5 = lbind(subtract, x3)
    x6 = compose(x4, size)
    x7 = chain(x4, x5, size)
    x8 = fork(canvas, color, x6)
    x9 = lbind(canvas, ZERO)
    x10 = compose(x9, x7)
    x11 = fork(vconcat, x8, x10)
    x12 = compose(cmirror, x11)
    x13 = apply(x12, x2)
    x14 = merge(x13)
    O = cmirror(x14)
    return O
