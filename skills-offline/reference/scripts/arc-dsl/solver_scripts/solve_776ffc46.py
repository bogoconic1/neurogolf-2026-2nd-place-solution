from dsl import *
from constants import *


def solve_776ffc46(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, FIVE)
    x3 = fork(equality, toindices, box)
    x4 = extract(x2, x3)
    x5 = inbox(x4)
    x6 = subgrid(x5, I)
    x7 = asobject(x6)
    x8 = matcher(first, ZERO)
    x9 = compose(flip, x8)
    x10 = sfilter(x7, x9)
    x11 = normalize(x10)
    x12 = toindices(x11)
    x13 = compose(toindices, normalize)
    x14 = matcher(x13, x12)
    x15 = mfilter(x1, x14)
    x16 = color(x11)
    O = fill(I, x16, x15)
    return O
