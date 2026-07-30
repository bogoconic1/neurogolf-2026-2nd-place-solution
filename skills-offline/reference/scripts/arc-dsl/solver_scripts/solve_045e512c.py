from dsl import *
from constants import *


def solve_045e512c(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    x3 = remove(x2, x1)
    x4 = lbind(shift, x2)
    x5 = lbind(mapply, x4)
    x6 = double(TEN)
    x7 = interval(FOUR, x6, FOUR)
    x8 = rbind(apply, x7)
    x9 = lbind(position, x2)
    x10 = lbind(rbind, multiply)
    x11 = chain(x8, x10, x9)
    x12 = compose(x5, x11)
    x13 = fork(recolor, color, x12)
    x14 = mapply(x13, x3)
    O = paint(I, x14)
    return O
