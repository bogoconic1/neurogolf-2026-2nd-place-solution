from dsl import *
from constants import *


def solve_2bcee788(I):
    x1 = mostcolor(I)
    x2 = objects(I, T, F, T)
    x3 = replace(I, x1, THREE)
    x4 = argmax(x2, size)
    x5 = argmin(x2, size)
    x6 = position(x4, x5)
    x7 = first(x6)
    x8 = last(x6)
    x9 = subgrid(x4, x3)
    x10 = hline(x5)
    x11 = hmirror(x9)
    x12 = vmirror(x9)
    x13 = branch(x10, x11, x12)
    x14 = branch(x10, x7, ZERO)
    x15 = branch(x10, ZERO, x8)
    x16 = asobject(x13)
    x17 = matcher(first, THREE)
    x18 = compose(flip, x17)
    x19 = sfilter(x16, x18)
    x20 = ulcorner(x4)
    x21 = shape(x4)
    x22 = astuple(x14, x15)
    x23 = multiply(x21, x22)
    x24 = add(x20, x23)
    x25 = shift(x19, x24)
    O = paint(x3, x25)
    return O
