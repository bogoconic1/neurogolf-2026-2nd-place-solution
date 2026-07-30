from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task080'
TASK_NUM = 80
KAGGLE = {'score': 19.038995, 'date': '2026-07-13'}
MEMORY_BYTES = 184
PARAMS = 204
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task080_stateful_bernoulli_bits'
OPSETS = [('', 20)]


# Cemb: FLOAT[10, 3], 30 value(s)
INIT_CEMB_0 = [0.9510565400123596, 0.30901700258255005, 1.0, 0.5877852439880371, 0.80901700258255, 1.0, -2.7866818186339515e-08, 1.0,
 1.0, -0.5877853035926819, 0.8090169429779053, 1.0, -0.9510565400123596, 0.30901697278022766, 1.0, -0.9510564804077148,
 -0.3090170919895172, 1.0, -0.5877851843833923, -0.80901700258255, 1.0, -1.3227050033037813e-07, -1.0, 1.0,
 0.5877853631973267, -0.8090169429779053, 1.0, 0.9510566592216492, -0.309016615152359, 1.0]

# T: FLOAT[2, 3, 3], 18 value(s)
INIT_T_1 = [50656.73046875, 1.832368781151672e-07, -8350.443359375, -0.02421184815466404, 153562.78125, -7705.4912109375,
 8721.369140625, 2802.709716796875, -45620.02734375, 167133.96875, 55762.5859375, 260939.984375, 175500.0625,
 52638.01953125, 253011.46875, -133757.9375, -41428.1484375, -218729.90625]

# Froot: FLOAT[30, 3], 90 value(s)
INIT_FROOT_2 = [0.1129937395453453, 0.1129937395453453, 0.0, 0.1129937395453453, 0.11106980592012405, 0.020762544125318527,
 0.1129937395453453, 0.10536352545022964, 0.040818046778440475, 0.1129937395453453, 0.09606920927762985,
 0.05948353931307793, 0.1129937395453453, 0.08350338041782379, 0.07612338662147522, 0.1129937395453453,
 0.06809394806623459, 0.0901709496974945, 0.1129937395453453, 0.050365641713142395, 0.1011478453874588,
 0.1129937395453453, 0.03092220425605774, 0.10868027806282043, 0.1129937395453453, 0.0104257483035326,
 0.11251172423362732, 0.1129937395453453, -0.010425744578242302, 0.11251172423362732, 0.1129937395453453,
 -0.030922196805477142, 0.10868027806282043, 0.1129937395453453, -0.0503656342625618, 0.1011478453874588,
 0.1129937395453453, -0.06809394806623459, 0.0901709496974945, 0.1129937395453453, -0.0835033729672432,
 0.07612340152263641, 0.1129937395453453, -0.09606920927762985, 0.05948353931307793, 0.1129937395453453,
 -0.10536351799964905, 0.04081806167960167, 0.1129937395453453, -0.11106980592012405, 0.020762545987963676,
 0.1129937395453453, -0.1129937395453453, -9.878226236992305e-09, 0.1129937395453453, -0.11106980592012405,
 -0.02076253853738308, 0.1129937395453453, -0.10536352545022964, -0.04081805422902107, 0.1129937395453453,
 -0.09606921672821045, -0.05948353186249733, 0.1129937395453453, -0.08350338041782379, -0.07612338662147522,
 0.1129937395453453, -0.06809394806623459, -0.09017094224691391, 0.1129937395453453, -0.050365645438432693,
 -0.1011478453874588, 0.1129937395453453, -0.030922193080186844, -0.10868027806282043, 0.1129937395453453,
 -0.010425725020468235, -0.11251173168420792, 0.1129937395453453, 0.010425727814435959, -0.11251173168420792,
 0.1129937395453453, 0.030922193080186844, -0.10868027806282043, 0.1129937395453453, 0.050365645438432693,
 -0.1011478453874588, 0.1129937395453453, 0.06809394806623459, -0.09017094224691391]

# RouteGeom: FLOAT[2, 2, 3], 12 value(s)
INIT_ROUTEGEOM_3 = [19053676544.0, 0.0, 0.0, 0.0, 19053676544.0, 0.0, 0.0, 19053676544.0, 0.0, 0.0, 0.0, 19053676544.0]

# Var0Lane: FLOAT[2, 3], 6 value(s)
INIT_VAR0LANE_4 = [-16622278410240.0, 19550626119680.0, 19550626119680.0, 10380823756800.0, -10380823756800.0, -10380823756800.0]

# Var1: FLOAT[2, 3, 2], 12 value(s)
INIT_VAR1_5 = [-0.29302701354026794, 0.3965134918689728, -0.29302701354026794, 0.3965134918689728, -0.29302701354026794,
 0.3965134918689728, -0.29302704334259033, 0.39651352167129517, 0.21222472190856934, -0.24961236119270325,
 0.314491868019104, -0.36989593505859375]

# Var2: FLOAT[3, 2, 2], 12 value(s)
INIT_VAR2_6 = [-1712938184343552.0, 2842415798943744.0, -6452091534966784.0, 1.070647300063232e+16, -1712938184343552.0,
 2842415798943744.0, -1.024925762584576e+16, 1.7007414435905536e+16, -1712938184343552.0, 2842415798943744.0,
 8481536596770816.0, -1.1476906381672448e+16]

# Common10: FLOAT[3], 3 value(s)
INIT_COMMON10_7 = [0.9324824213981628, 1.0, 1.0]

# Common11: FLOAT[3], 3 value(s)
INIT_COMMON11_8 = [0.994294285774231, 1.0, 1.0]

# RouteType: FLOAT[3, 2], 6 value(s)
INIT_ROUTETYPE_9 = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]

# src_false: FLOAT[2, 1], 2 value(s)
INIT_SRC_FALSE_10 = [0.5, 0.0]

# SignP: FLOAT[3], 3 value(s)
INIT_SIGNP_11 = [1.0, -1.0, -1.0]

# src_true: FLOAT[1], 1 value(s)
INIT_SRC_TRUE_12 = [0.5]

# u8_0: UINT8[1], 1 value(s)
INIT_U8_0_13 = [0]

# u8_4: UINT8[1], 1 value(s)
INIT_U8_4_14 = [4]

# u8_8: UINT8[1], 1 value(s)
INIT_U8_8_15 = [8]

# rng_p45: FLOAT[1], 1 value(s)
INIT_RNG_P45_16 = [0.44999998807907104]

# rng_p65: FLOAT[1], 1 value(s)
INIT_RNG_P65_17 = [0.6499999761581421]

# false_scalar: BOOL[1], 1 value(s)
INIT_FALSE_SCALAR_18 = [False]


NP_DTYPE = {
    TensorProto.FLOAT: np.float32,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT8: np.int8,
    TensorProto.UINT16: np.uint16,
    TensorProto.INT16: np.int16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.BOOL: np.bool_,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
    TensorProto.UINT32: np.uint32,
    TensorProto.UINT64: np.uint64,
}


def _num(value):
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    if value == "nan":
        return float("nan")
    return value


def _tensor(name: str, elem_type: int, shape: tuple[int, ...], values: list) -> onnx.TensorProto:
    if elem_type == TensorProto.STRING:
        vals = [v.encode("utf-8") if isinstance(v, str) else v for v in values]
        return helper.make_tensor(name=name, data_type=TensorProto.STRING, dims=list(shape), vals=vals)
    flat = [_num(value) for value in values]
    array = np.asarray(flat, dtype=NP_DTYPE[elem_type]).reshape(shape)
    return numpy_helper.from_array(array, name=name)


def _vi(name: str, elem_type: int, shape: list[int | str | None]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


class _EmptyAttr:
    # Sentinel for an EMPTY list-valued attribute (INTS/FLOATS/STRINGS, e.g. a scalar
    # RandomUniform's shape=[]). helper.make_node() cannot infer the attribute type from a
    # bare [], so _node() re-adds these with an explicit attr_type after building the node.
    __slots__ = ("kind",)

    def __init__(self, kind: str) -> None:
        self.kind = kind


def _node(op_type, inputs, outputs, name="", domain="", **attrs):
    empties = [(k, v.kind) for k, v in attrs.items() if isinstance(v, _EmptyAttr)]
    for k, _ in empties:
        attrs.pop(k)
    node = helper.make_node(op_type, inputs, outputs, name=name, domain=domain, **attrs)
    for k, kind in empties:
        node.attribute.append(helper.make_attribute(k, [], attr_type=getattr(AttributeProto, kind)))
    return node


def make_onnx() -> onnx.ModelProto:
    nodes = [
        # 0: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['target_bit1'],
            name='',
            domain='',
            dtype=9,
            seed=1062279.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['target_bit2'],
            name='',
            domain='',
            dtype=9,
            seed=368587424.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p45'],
            ['role1_bit2'],
            name='',
            domain='',
            dtype=9,
            seed=230791472.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['role1_bit3'],
            name='',
            domain='',
            dtype=9,
            seed=123758288.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['role2_bit2'],
            name='',
            domain='',
            dtype=9,
            seed=58562012.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['role2_bit3'],
            name='',
            domain='',
            dtype=9,
            seed=2205103.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['zsrc_1'],
            name='',
            domain='',
            dtype=9,
            seed=8337775.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['zsrc_2'],
            name='',
            domain='',
            dtype=9,
            seed=95380048.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['zsrc_4'],
            name='',
            domain='',
            dtype=9,
            seed=5674635.0,
        ),
        # 9: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['zsrc_5'],
            name='',
            domain='',
            dtype=9,
            seed=2508287.0,
        ),
        # 10: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['zsrc_6'],
            name='',
            domain='',
            dtype=9,
            seed=349634.0,
        ),
        # 11: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['zsrc_8'],
            name='',
            domain='',
            dtype=9,
            seed=4897278.0,
        ),
        # 12: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['zsrc_9'],
            name='',
            domain='',
            dtype=9,
            seed=5238709.0,
        ),
        # 13: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p45'],
            ['role1_term0'],
            name='',
            domain='',
            dtype=2,
            seed=52875308.0,
        ),
        # 14: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['role2_term0'],
            name='',
            domain='',
            dtype=2,
            seed=99418424.0,
        ),
        # 15: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['src_true'],
            ['role1_bit1_u8'],
            name='',
            domain='',
            dtype=2,
            seed=38172900.0,
        ),
        # 16: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p65'],
            ['role2_bit1_u8'],
            name='',
            domain='',
            dtype=2,
            seed=768920.0,
        ),
        # 17: Add inputs=2 outputs=1
        _node(
            'Add',
            ['role1_bit1_u8', 'role1_bit1_u8'],
            ['role1_term1'],
            name='',
            domain='',
        ),
        # 18: Add inputs=2 outputs=1
        _node(
            'Add',
            ['role2_bit1_u8', 'role2_bit1_u8'],
            ['role2_term1'],
            name='',
            domain='',
        ),
        # 19: Equal inputs=2 outputs=1
        _node(
            'Equal',
            ['target_bit1', 'target_bit2'],
            ['target_bit0'],
            name='',
            domain='',
        ),
        # 20: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['target_bit0', 'target_bit1', 'target_bit2'],
            ['target_basis_bool'],
            name='',
            domain='',
            axis=0,
        ),
        # 21: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['target_basis_bool'],
            ['target_basis'],
            name='',
            domain='',
            to=1,
        ),
        # 22: Where inputs=3 outputs=1
        _node(
            'Where',
            ['role1_bit2', 'u8_4', 'u8_0'],
            ['role1_term2'],
            name='',
            domain='',
        ),
        # 23: Add inputs=2 outputs=1
        _node(
            'Add',
            ['role1_term0', 'role1_term1'],
            ['role1_sum01'],
            name='',
            domain='',
        ),
        # 24: Where inputs=3 outputs=1
        _node(
            'Where',
            ['role1_bit3', 'u8_8', 'role1_term2'],
            ['role1_sum23'],
            name='',
            domain='',
        ),
        # 25: Add inputs=2 outputs=1
        _node(
            'Add',
            ['role1_sum01', 'role1_sum23'],
            ['role1_idxu'],
            name='',
            domain='',
        ),
        # 26: Where inputs=3 outputs=1
        _node(
            'Where',
            ['role2_bit2', 'u8_4', 'u8_0'],
            ['role2_term2'],
            name='',
            domain='',
        ),
        # 27: Add inputs=2 outputs=1
        _node(
            'Add',
            ['role2_term0', 'role2_term1'],
            ['role2_sum01'],
            name='',
            domain='',
        ),
        # 28: Where inputs=3 outputs=1
        _node(
            'Where',
            ['role2_bit3', 'u8_8', 'role2_term2'],
            ['role2_sum23'],
            name='',
            domain='',
        ),
        # 29: Add inputs=2 outputs=1
        _node(
            'Add',
            ['role2_sum01', 'role2_sum23'],
            ['role2_idxu'],
            name='',
            domain='',
        ),
        # 30: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['u8_0', 'role1_idxu', 'role2_idxu'],
            ['role_idx_u8'],
            name='',
            domain='',
            axis=0,
        ),
        # 31: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['role_idx_u8'],
            ['role_idx'],
            name='',
            domain='',
            to=6,
        ),
        # 32: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['Cemb', 'role_idx'],
            ['role_comp'],
            name='',
            domain='',
            axis=0,
        ),
        # 33: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['false_scalar', 'zsrc_1', 'zsrc_2', 'zsrc_8', 'zsrc_4', 'zsrc_5', 'zsrc_6', 'zsrc_9', 'zsrc_8',
             'zsrc_9'],
            ['Zsrc'],
            name='',
            domain='',
            axis=0,
        ),
        # 34: Where inputs=3 outputs=1
        _node(
            'Where',
            ['Zsrc', 'src_true', 'src_false'],
            ['source_stack'],
            name='',
            domain='',
        ),
        # 35: Einsum inputs=760 outputs=1
        _node(
            'Einsum',
            ['SignP', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11',
             'Common11', 'Common11', 'Froot', 'Froot', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP',
             'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'Common10', 'Common10',
             'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11',
             'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11',
             'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10',
             'Common10', 'Common11', 'Common11', 'Common11', 'Froot', 'Froot', 'Froot', 'Froot', 'Froot',
             'Froot', 'SignP', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11',
             'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11',
             'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10',
             'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common10',
             'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Froot',
             'Froot', 'Froot', 'Common11', 'Common11', 'Common11', 'Froot', 'SignP', 'Common10', 'SignP',
             'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10',
             'Common10', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11',
             'Common11', 'Common11', 'Common10', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP',
             'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'Common10', 'Common11',
             'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11',
             'Common10', 'Common10', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11',
             'Common10', 'Common10', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11',
             'Common10', 'Common10', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11',
             'Froot', 'Froot', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP',
             'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'Common10', 'Common11', 'Common11',
             'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10',
             'Common10', 'Froot', 'Froot', 'SignP', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10',
             'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11',
             'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11',
             'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10',
             'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP',
             'Common10', 'SignP', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common10',
             'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common10',
             'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common10',
             'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common10',
             'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Froot', 'Froot', 'Froot', 'SignP',
             'Common10', 'Froot', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP',
             'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common11', 'Common11', 'Common11',
             'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10',
             'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11',
             'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11',
             'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common11', 'Common11',
             'Common11', 'Common11', 'Common11', 'Common11', 'Common11', 'Common11', 'Froot', 'Froot', 'Froot',
             'Froot', 'Common10', 'Froot', 'Froot', 'Var0Lane', 'Froot', 'Froot', 'RouteType', 'Froot', 'Froot',
             'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP',
             'Common10', 'SignP', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Froot', 'Froot',
             'RouteType', 'Var2', 'Var1', 'Froot', 'Froot', 'target_basis', 'input', 'source_stack', 'Var1',
             'Var2', 'RouteGeom', 'RouteType', 'role_comp', 'Var0Lane', 'Froot', 'T', 'Cemb', 'input', 'Froot',
             'RouteType', 'Froot', 'Froot', 'RouteType', 'Froot', 'Froot', 'Froot', 'Common10', 'Froot',
             'Froot', 'SignP', 'Common11', 'Common11', 'Common11', 'Froot', 'Froot', 'Common11', 'Common11',
             'Common11', 'Froot', 'Froot', 'SignP', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10',
             'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11',
             'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11',
             'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10',
             'Froot', 'Froot', 'Froot', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11',
             'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP',
             'Common10', 'SignP', 'Common10', 'SignP', 'Common11', 'Common11', 'Common11', 'Common10',
             'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11',
             'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11',
             'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10',
             'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common11', 'Common11', 'Common11',
             'Common11', 'Common11', 'Common11', 'Common11', 'Common11', 'Froot', 'Froot', 'SignP', 'Common10',
             'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP',
             'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11',
             'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Froot', 'Froot', 'SignP', 'Common10',
             'Froot', 'Froot', 'Froot', 'Froot', 'Froot', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP',
             'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'Common10', 'Common11',
             'Common11', 'Common11', 'Froot', 'Froot', 'Froot', 'Common10', 'Common10', 'Common11', 'Common11',
             'Common11', 'Common11', 'Common10', 'Common10', 'Froot', 'Froot', 'Froot', 'SignP', 'Common10',
             'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP',
             'Common10', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10',
             'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11',
             'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11',
             'Common11', 'Common10', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Froot',
             'SignP', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11',
             'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11',
             'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10',
             'Common11', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'SignP', 'Common10',
             'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP',
             'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11',
             'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common10', 'Common10', 'Common11',
             'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common10', 'Common10', 'Common11',
             'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common10', 'Common10', 'Common11',
             'Common11', 'Common11', 'Common11', 'Froot', 'Froot', 'SignP', 'Common10', 'SignP', 'Common10',
             'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'Common10',
             'Common11', 'Common11', 'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11',
             'Common11', 'Common10', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10',
             'SignP', 'Common10', 'SignP', 'Common10', 'SignP', 'Common10', 'Common10', 'Common11', 'Common11',
             'Common11', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10',
             'Common10', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10',
             'Common10', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Common10',
             'Common10', 'Common10', 'Common10', 'Common11', 'Common11', 'Common11', 'Common11', 'Cemb'],
            ['output'],
            name='',
            domain='',
            equation='v,v,v,v,F,F,F,F,F,F,gF,gv,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,gB,cF,cB,cv,gx,cx,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,E,E,E,E,E,E,E,E,cE,gE,cH,H,H,H,gH,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,cA,gA,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,cC,gC,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,gz,cz,gw,w,w,cw,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,cy,gy,cG,gG,G,gs,cs,is,cu,gu,uo,gD,cD,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,gt,ct,ta,lio,ila,gI,cI,l,...fgh,nf,jlk,ljm,ijp,pn,pq,jJ,hJ,nqr,er,...ecd,dJ,Kk,hK,dK,Lm,dL,hL,dX,X,hX,dM,M,M,M,M,hM,dY,Y,Y,Y,hY,hO,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,dO,hW,dW,W,W,W,W,W,W,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,hP,dP,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,hT,dT,N,N,hN,dN,hZ,dZ,hU,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,dU,hV,dV,V,V,V,V,V,V,V,V,hR,hQ,hS,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,dS,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,dQ,dR,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,bq->...bcd',
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('Cemb', TensorProto.FLOAT, (10, 3), INIT_CEMB_0),
            _tensor('T', TensorProto.FLOAT, (2, 3, 3), INIT_T_1),
            _tensor('Froot', TensorProto.FLOAT, (30, 3), INIT_FROOT_2),
            _tensor('RouteGeom', TensorProto.FLOAT, (2, 2, 3), INIT_ROUTEGEOM_3),
            _tensor('Var0Lane', TensorProto.FLOAT, (2, 3), INIT_VAR0LANE_4),
            _tensor('Var1', TensorProto.FLOAT, (2, 3, 2), INIT_VAR1_5),
            _tensor('Var2', TensorProto.FLOAT, (3, 2, 2), INIT_VAR2_6),
            _tensor('Common10', TensorProto.FLOAT, (3,), INIT_COMMON10_7),
            _tensor('Common11', TensorProto.FLOAT, (3,), INIT_COMMON11_8),
            _tensor('RouteType', TensorProto.FLOAT, (3, 2), INIT_ROUTETYPE_9),
            _tensor('src_false', TensorProto.FLOAT, (2, 1), INIT_SRC_FALSE_10),
            _tensor('SignP', TensorProto.FLOAT, (3,), INIT_SIGNP_11),
            _tensor('src_true', TensorProto.FLOAT, (1,), INIT_SRC_TRUE_12),
            _tensor('u8_0', TensorProto.UINT8, (1,), INIT_U8_0_13),
            _tensor('u8_4', TensorProto.UINT8, (1,), INIT_U8_4_14),
            _tensor('u8_8', TensorProto.UINT8, (1,), INIT_U8_8_15),
            _tensor('rng_p45', TensorProto.FLOAT, (1,), INIT_RNG_P45_16),
            _tensor('rng_p65', TensorProto.FLOAT, (1,), INIT_RNG_P65_17),
            _tensor('false_scalar', TensorProto.BOOL, (1,), INIT_FALSE_SCALAR_18),
        ],
        value_info=[

        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(domain, version) for domain, version in OPSETS],
        producer_name=PRODUCER_NAME,
        producer_version=PRODUCER_VERSION,
        domain=DOMAIN,
        model_version=MODEL_VERSION,
        doc_string="",
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def save_model(path: str | Path = OUT) -> None:
    onnx.save_model(make_onnx(), path)


if __name__ == "__main__":
    save_model(sys.argv[1] if len(sys.argv) > 1 else OUT)
