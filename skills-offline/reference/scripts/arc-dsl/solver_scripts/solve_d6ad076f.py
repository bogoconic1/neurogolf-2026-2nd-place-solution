from dsl import *
from constants import *


def solve_d6ad076f(I):
    x1 = objects(I, T, F, T)
    x2 = argmin(x1, size)
    x3 = argmax(x1, size)
    x4 = vmatching(x2, x3)
    x5 = branch(x4, DOWN, RIGHT)
    x6 = branch(x4, uppermost, leftmost)
    x7 = valmax(x1, x6)
    x8 = x6(x2)
    x9 = equality(x7, x8)
    x10 = branch(x9, NEG_ONE, ONE)
    x11 = multiply(x5, x10)
    x12 = inbox(x2)
    x13 = rbind(shoot, x11)
    x14 = mapply(x13, x12)
    x15 = underfill(I, EIGHT, x14)
    x16 = objects(x15, T, F, T)
    x17 = colorfilter(x16, EIGHT)
    x18 = rbind(bordering, I)
    x19 = mfilter(x17, x18)
    O = cover(x15, x19)
    return O
