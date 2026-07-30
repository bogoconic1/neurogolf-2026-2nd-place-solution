from dsl import *
from constants import *


def solve_3eda0437(I):
    x1 = interval(TWO, TEN, ONE)
    x2 = prapply(astuple, x1, x1)
    x3 = lbind(canvas, ZERO)
    x4 = lbind(occurrences, I)
    x5 = lbind(lbind, shift)
    x6 = fork(apply, x5, x4)
    x7 = chain(x6, asobject, x3)
    x8 = mapply(x7, x2)
    x9 = argmax(x8, size)
    O = fill(I, SIX, x9)
    return O
