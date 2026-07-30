from dsl import *
from constants import *


def solve_82819916(I):
    x1 = objects(I, F, T, T)
    x2 = argmax(x1, size)
    x3 = remove(x2, x1)
    x4 = normalize(x2)
    x5 = compose(last, last)
    x6 = rbind(argmin, x5)
    x7 = compose(first, x6)
    x8 = fork(other, palette, x7)
    x9 = x7(x4)
    x10 = matcher(first, x9)
    x11 = sfilter(x4, x10)
    x12 = difference(x4, x11)
    x13 = compose(toivec, uppermost)
    x14 = lbind(shift, x11)
    x15 = lbind(shift, x12)
    x16 = compose(x14, x13)
    x17 = compose(x15, x13)
    x18 = fork(recolor, x7, x16)
    x19 = fork(recolor, x8, x17)
    x20 = fork(combine, x18, x19)
    x21 = mapply(x20, x3)
    O = paint(I, x21)
    return O
