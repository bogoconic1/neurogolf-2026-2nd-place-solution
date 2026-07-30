from dsl import *
from constants import *


def solve_b2862040(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, NINE)
    x3 = colorfilter(x1, ONE)
    x4 = rbind(bordering, I)
    x5 = compose(flip, x4)
    x6 = mfilter(x2, x5)
    x7 = rbind(adjacent, x6)
    x8 = mfilter(x3, x7)
    O = fill(I, EIGHT, x8)
    return O
