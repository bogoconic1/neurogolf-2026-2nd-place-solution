from dsl import *
from constants import *


def solve_22233c11(I):
    x1 = objects(I, T, T, T)
    x2 = rbind(upscale, TWO)
    x3 = chain(invert, halve, shape)
    x4 = fork(combine, hfrontier, vfrontier)
    x5 = compose(x2, vmirror)
    x6 = fork(shift, x5, x3)
    x7 = compose(toindices, x6)
    x8 = lbind(mapply, x4)
    x9 = compose(x8, toindices)
    x10 = fork(difference, x7, x9)
    x11 = mapply(x10, x1)
    O = fill(I, EIGHT, x11)
    return O
