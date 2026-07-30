from dsl import *
from constants import *


def solve_810b9b61(I):
    x1 = objects(I, T, T, T)
    x2 = apply(toindices, x1)
    x3 = fork(either, vline, hline)
    x4 = sfilter(x2, x3)
    x5 = difference(x2, x4)
    x6 = fork(equality, identity, box)
    x7 = mfilter(x5, x6)
    O = fill(I, THREE, x7)
    return O
