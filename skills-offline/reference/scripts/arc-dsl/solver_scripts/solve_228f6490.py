from dsl import *
from constants import *


def solve_228f6490(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = sfilter(x2, x4)
    x6 = first(x5)
    x7 = last(x5)
    x8 = difference(x1, x2)
    x9 = compose(normalize, toindices)
    x10 = x9(x6)
    x11 = x9(x7)
    x12 = matcher(x9, x10)
    x13 = matcher(x9, x11)
    x14 = extract(x8, x12)
    x15 = extract(x8, x13)
    x16 = ulcorner(x6)
    x17 = ulcorner(x7)
    x18 = ulcorner(x14)
    x19 = ulcorner(x15)
    x20 = subtract(x16, x18)
    x21 = subtract(x17, x19)
    x22 = move(I, x14, x20)
    O = move(x22, x15, x21)
    return O
