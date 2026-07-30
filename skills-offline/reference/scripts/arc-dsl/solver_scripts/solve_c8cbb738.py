from dsl import *
from constants import *


def solve_c8cbb738(I):
    x1 = mostcolor(I)
    x2 = fgpartition(I)
    x3 = valmax(x2, shape)
    x4 = canvas(x1, x3)
    x5 = apply(normalize, x2)
    x6 = lbind(subtract, x3)
    x7 = chain(halve, x6, shape)
    x8 = fork(shift, identity, x7)
    x9 = mapply(x8, x5)
    O = paint(x4, x9)
    return O
