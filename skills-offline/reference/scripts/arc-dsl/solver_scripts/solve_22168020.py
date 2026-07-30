from dsl import *
from constants import *


def solve_22168020(I):
    x1 = palette(I)
    x2 = remove(ZERO, x1)
    x3 = lbind(ofcolor, I)
    x4 = lbind(prapply, connect)
    x5 = fork(x4, x3, x3)
    x6 = compose(merge, x5)
    x7 = fork(recolor, identity, x6)
    x8 = mapply(x7, x2)
    O = paint(I, x8)
    return O
