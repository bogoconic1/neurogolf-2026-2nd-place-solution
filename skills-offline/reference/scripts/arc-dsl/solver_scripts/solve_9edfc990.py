from dsl import *
from constants import *


def solve_9edfc990(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = ofcolor(I, ONE)
    x4 = rbind(adjacent, x3)
    x5 = mfilter(x2, x4)
    x6 = recolor(ONE, x5)
    O = paint(I, x6)
    return O
