from dsl import *
from constants import *


def solve_3906de3d(I):
    x1 = rot270(I)
    x2 = rbind(order, identity)
    x3 = switch(x1, ONE, TWO)
    x4 = apply(x2, x3)
    x5 = switch(x4, ONE, TWO)
    O = cmirror(x5)
    return O
