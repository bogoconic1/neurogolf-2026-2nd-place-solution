from dsl import *
from constants import *


def solve_fcc82909(I):
    x1 = objects(I, F, T, T)
    x2 = rbind(add, DOWN)
    x3 = compose(x2, llcorner)
    x4 = compose(toivec, numcolors)
    x5 = fork(add, lrcorner, x4)
    x6 = fork(astuple, x3, x5)
    x7 = compose(box, x6)
    x8 = mapply(x7, x1)
    O = fill(I, THREE, x8)
    return O
