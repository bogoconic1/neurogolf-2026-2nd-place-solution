from dsl import *
from constants import *


def solve_a8c38be5(I):
    x1 = replace(I, FIVE, ZERO)
    x2 = objects(x1, T, F, T)
    x3 = apply(normalize, x2)
    x4 = astuple(NINE, NINE)
    x5 = canvas(FIVE, x4)
    x6 = asindices(x5)
    x7 = box(x6)
    x8 = center(x6)
    x9 = lbind(contained, ZERO)
    x10 = rbind(subtract, x8)
    x11 = compose(x9, x10)
    x12 = chain(outbox, outbox, initset)
    x13 = corners(x6)
    x14 = mapply(x12, x13)
    x15 = difference(x7, x14)
    x16 = inbox(x7)
    x17 = sfilter(x16, x11)
    x18 = combine(x15, x17)
    x19 = fill(x5, ONE, x18)
    x20 = objects(x19, T, F, T)
    x21 = apply(toindices, x20)
    x22 = lbind(matcher, normalize)
    x23 = lbind(extract, x21)
    x24 = chain(ulcorner, x23, x22)
    x25 = compose(x24, toindices)
    x26 = fork(shift, identity, x25)
    x27 = mapply(x26, x3)
    O = paint(x5, x27)
    return O
