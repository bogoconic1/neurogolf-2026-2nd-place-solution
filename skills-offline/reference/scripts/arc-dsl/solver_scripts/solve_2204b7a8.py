from dsl import *
from constants import *


def solve_2204b7a8(I):
    x1 = objects(I, T, F, T)
    x2 = lbind(sfilter, x1)
    x3 = compose(size, x2)
    x4 = x3(vline)
    x5 = x3(hline)
    x6 = greater(x4, x5)
    x7 = branch(x6, lefthalf, tophalf)
    x8 = branch(x6, righthalf, bottomhalf)
    x9 = branch(x6, hconcat, vconcat)
    x10 = x7(I)
    x11 = x8(I)
    x12 = index(x10, ORIGIN)
    x13 = shape(x11)
    x14 = decrement(x13)
    x15 = index(x11, x14)
    x16 = replace(x10, THREE, x12)
    x17 = replace(x11, THREE, x15)
    O = x9(x16, x17)
    return O
