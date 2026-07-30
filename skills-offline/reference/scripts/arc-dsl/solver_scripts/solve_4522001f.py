from dsl import *
from constants import *


def solve_4522001f(I):
    x1 = objects(I, F, F, T)
    x2 = first(x1)
    x3 = toindices(x2)
    x4 = contained(ZERO_BY_TWO, x3)
    x5 = contained(TWO_BY_TWO, x3)
    x6 = contained(TWO_BY_ZERO, x3)
    x7 = astuple(NINE, NINE)
    x8 = canvas(ZERO, x7)
    x9 = astuple(THREE, ORIGIN)
    x10 = initset(x9)
    x11 = upscale(x10, TWO)
    x12 = upscale(x11, TWO)
    x13 = shape(x12)
    x14 = shift(x12, x13)
    x15 = combine(x12, x14)
    x16 = paint(x8, x15)
    x17 = rot90(x16)
    x18 = rot180(x16)
    x19 = rot270(x16)
    x20 = branch(x4, x17, x16)
    x21 = branch(x5, x18, x20)
    O = branch(x6, x19, x21)
    return O
