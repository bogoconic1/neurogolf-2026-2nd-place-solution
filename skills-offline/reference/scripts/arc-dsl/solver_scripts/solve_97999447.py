from dsl import *
from constants import *


def solve_97999447(I):
    x1 = objects(I, T, F, T)
    x2 = apply(toindices, x1)
    x3 = rbind(shoot, RIGHT)
    x4 = compose(x3, center)
    x5 = fork(recolor, color, x4)
    x6 = mapply(x5, x1)
    x7 = paint(I, x6)
    x8 = interval(ZERO, FIVE, ONE)
    x9 = apply(double, x8)
    x10 = apply(increment, x9)
    x11 = apply(tojvec, x10)
    x12 = prapply(shift, x2, x11)
    x13 = merge(x12)
    O = fill(x7, FIVE, x13)
    return O
