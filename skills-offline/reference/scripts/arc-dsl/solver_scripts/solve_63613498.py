from dsl import *
from constants import *


def solve_63613498(I):
    x1 = crop(I, ORIGIN, THREE_BY_THREE)
    x2 = ofcolor(x1, ZERO)
    x3 = asindices(x1)
    x4 = difference(x3, x2)
    x5 = normalize(x4)
    x6 = objects(I, T, F, T)
    x7 = compose(toindices, normalize)
    x8 = matcher(x7, x5)
    x9 = mfilter(x6, x8)
    x10 = fill(I, FIVE, x9)
    x11 = asobject(x1)
    O = paint(x10, x11)
    return O
