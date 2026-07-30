from dsl import *
from constants import *


def solve_5daaa586(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = extract(x2, x4)
    x6 = outbox(x5)
    x7 = subgrid(x6, I)
    x8 = fgpartition(x7)
    x9 = argmax(x8, size)
    x10 = color(x9)
    x11 = toindices(x9)
    x12 = prapply(connect, x11, x11)
    x13 = mfilter(x12, vline)
    x14 = mfilter(x12, hline)
    x15 = size(x13)
    x16 = size(x14)
    x17 = greater(x15, x16)
    x18 = branch(x17, x13, x14)
    O = fill(x7, x10, x18)
    return O
