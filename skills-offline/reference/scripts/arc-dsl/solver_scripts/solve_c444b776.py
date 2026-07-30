from dsl import *
from constants import *


def solve_c444b776(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = argmin(x2, size)
    x4 = backdrop(x3)
    x5 = toobject(x4, I)
    x6 = normalize(x5)
    x7 = lbind(shift, x6)
    x8 = compose(x7, ulcorner)
    x9 = mapply(x8, x2)
    O = paint(I, x9)
    return O
