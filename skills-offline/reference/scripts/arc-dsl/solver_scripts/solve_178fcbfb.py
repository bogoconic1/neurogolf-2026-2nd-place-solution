from dsl import *
from constants import *


def solve_178fcbfb(I):
    x1 = objects(I, T, F, T)
    x2 = ofcolor(I, TWO)
    x3 = mapply(vfrontier, x2)
    x4 = fill(I, TWO, x3)
    x5 = colorfilter(x1, TWO)
    x6 = difference(x1, x5)
    x7 = compose(hfrontier, center)
    x8 = fork(recolor, color, x7)
    x9 = mapply(x8, x6)
    O = paint(x4, x9)
    return O
