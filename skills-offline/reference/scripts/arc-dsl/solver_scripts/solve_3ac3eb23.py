from dsl import *
from constants import *


def solve_3ac3eb23(I):
    x1 = objects(I, T, F, T)
    x2 = chain(ineighbors, last, first)
    x3 = fork(recolor, color, x2)
    x4 = mapply(x3, x1)
    x5 = paint(I, x4)
    x6 = vsplit(x5, THREE)
    x7 = first(x6)
    x8 = vconcat(x7, x7)
    O = vconcat(x7, x8)
    return O
