from dsl import *
from constants import *


def solve_56dc2b01(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, THREE)
    x3 = first(x2)
    x4 = ofcolor(I, TWO)
    x5 = gravitate(x3, x4)
    x6 = first(x5)
    x7 = equality(x6, ZERO)
    x8 = branch(x7, width, height)
    x9 = x8(x3)
    x10 = gravitate(x4, x3)
    x11 = sign(x10)
    x12 = multiply(x11, x9)
    x13 = crement(x12)
    x14 = recolor(EIGHT, x4)
    x15 = shift(x14, x13)
    x16 = paint(I, x15)
    O = move(x16, x3, x5)
    return O
