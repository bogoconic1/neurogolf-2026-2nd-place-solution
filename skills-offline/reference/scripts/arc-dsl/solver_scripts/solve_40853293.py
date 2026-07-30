from dsl import *
from constants import *


def solve_40853293(I):
    x1 = partition(I)
    x2 = fork(recolor, color, backdrop)
    x3 = apply(x2, x1)
    x4 = mfilter(x3, hline)
    x5 = mfilter(x3, vline)
    x6 = paint(I, x4)
    O = paint(x6, x5)
    return O
