from dsl import *
from constants import *


def solve_673ef223(I):
    x1 = objects(I, T, F, T)
    x2 = ofcolor(I, EIGHT)
    x3 = replace(I, EIGHT, FOUR)
    x4 = colorfilter(x1, TWO)
    x5 = argmin(x1, uppermost)
    x6 = apply(uppermost, x4)
    x7 = fork(subtract, maximum, minimum)
    x8 = x7(x6)
    x9 = toivec(x8)
    x10 = leftmost(x5)
    x11 = equality(x10, ZERO)
    x12 = branch(x11, LEFT, RIGHT)
    x13 = rbind(shoot, x12)
    x14 = mapply(x13, x2)
    x15 = underfill(x3, EIGHT, x14)
    x16 = shift(x2, x9)
    x17 = mapply(hfrontier, x16)
    O = underfill(x15, EIGHT, x17)
    return O
