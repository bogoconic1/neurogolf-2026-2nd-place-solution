from dsl import *
from constants import *


def solve_e40b9e2f(I):
    x1 = objects(I, F, T, T)
    x2 = neighbors(ORIGIN)
    x3 = mapply(neighbors, x2)
    x4 = first(x1)
    x5 = lbind(intersection, x4)
    x6 = compose(hmirror, vmirror)
    x7 = x6(x4)
    x8 = lbind(shift, x7)
    x9 = apply(x8, x3)
    x10 = argmax(x9, x5)
    x11 = paint(I, x10)
    x12 = objects(x11, F, T, T)
    x13 = first(x12)
    x14 = compose(size, x5)
    x15 = compose(vmirror, dmirror)
    x16 = x15(x13)
    x17 = lbind(shift, x16)
    x18 = apply(x17, x3)
    x19 = argmax(x18, x14)
    O = paint(x11, x19)
    return O
