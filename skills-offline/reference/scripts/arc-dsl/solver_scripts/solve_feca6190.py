from dsl import *
from constants import *


def solve_feca6190(I):
    x1 = objects(I, T, F, T)
    x2 = size(x1)
    x3 = multiply(x2, FIVE)
    x4 = astuple(x3, x3)
    x5 = canvas(ZERO, x4)
    x6 = rbind(shoot, UNITY)
    x7 = compose(x6, center)
    x8 = fork(recolor, color, x7)
    x9 = mapply(x8, x1)
    x10 = paint(x5, x9)
    O = hmirror(x10)
    return O
