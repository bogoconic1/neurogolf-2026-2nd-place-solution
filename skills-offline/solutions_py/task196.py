from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task196'
TASK_NUM = 196
KAGGLE = {'score': 19.943754, 'date': '2026-07-14'}
MEMORY_BYTES = 38
PARAMS = 119
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'intC_rulike196'
OPSETS = [('', 20)]


# like: FLOAT16[1, 3], 3 value(s)
INIT_LIKE_0 = [0.0, 0.0, 0.0]

# B: FLOAT[2, 30], 60 value(s)
INIT_B_1 = [4.396215438842773, 4.587011337280273, 3.7946882247924805, 5.643332004547119, 6.932193279266357, 6.494115352630615,
 5.236809253692627, 1.0995945930480957, 1.1569076776504517, 5.4401750564575195, 4.652101993560791, 3.6589348316192627,
 3.8211255073547363, 3.435670852661133, 6.076672077178955, 9.641462123553237e-20, 2.586544176928307e-20,
 4.811408757982598e-23, 3.417811546383487e-19, 6.210258781546458e-21, 2.1298252874363892e-21, 1.2362925112943911e-21,
 1.1377794604761953e-21, 3.279428947417796e-20, 2.711299330731356e-20, 1.829190169264721e-21, 1.7650388392488957e-21,
 5.809793505145688e-19, 1.8563876194089426e-22, 7.946862075548493e-22, -2.8827435970306396, -2.4082298278808594,
 -1.60280179977417, -2.3748013973236084, -1.0777565240859985, -0.5063771605491638, -0.019905248656868935,
 0.3130604326725006, 0.38222938776016235, 2.3074607849121094, 3.5356082916259766, 3.158010959625244, 3.3238298892974854,
 2.9890730381011963, 5.5186591148376465, 2.1624771535965307e-36, 1.3081502597935438e-18, 2.81776281467672e-34,
 8.096565112778021e-25, 2.8708150611488225e-30, -6.736390782577898e-35, 1.662555365225387e-27, 2.4635904431692268e-21,
 1.752722397803742e-27, 5.2274095310358075e-36, 3.4473414829079995e-24, 7.328424408372022e-23, 3.1523313180939174e-28,
 -6.420362657134927e-21, 4.7642298162753973e-35]

# U3: FLOAT[5, 3], 15 value(s)
INIT_U3_2 = [2.2814786434173584, -0.8171543478965759, 1.2038301229476929, -1.3150522708892822, 5.039955139160156,
 -0.3417642116546631, 0.8284443616867065, 4.294961929321289, -0.608856201171875, -1.4723304510116577,
 3.2520925998687744, 0.0227495227009058, -0.8113434910774231, -2.648597240447998, 1.3942677974700928]

# P: FLOAT[3, 2], 6 value(s)
INIT_P_3 = [0.999799907207489, 0.011445224285125732, 0.0031418476719409227, 0.945206880569458, 0.9775785207748413,
 0.9843820333480835]

# V3: FLOAT[5, 3], 15 value(s)
INIT_V3_4 = [2.3393354415893555, 1.8032376766204834, 0.3096776604652405, -0.35911181569099426, -4.099515914916992,
 0.9167236685752869, -1.2834810018539429, 3.0790064334869385, 0.4023405611515045, 2.1038870811462402,
 -0.6806387901306152, -1.7353878021240234, 1.1975961923599243, 2.878612995147705, 1.5011096000671387]

# E: FLOAT[10, 2], 20 value(s)
INIT_E_5 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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
        # 0: RandomUniformLike inputs=1 outputs=1
        _node(
            'RandomUniformLike',
            ['like'],
            ['rf'],
            name='',
            domain='',
            high=2048.0,
            low=0.0,
            seed=1.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['rf'],
            ['tok'],
            name='',
            domain='',
            to=6,
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['c'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=2,
            min_gram_length=1,
            mode='TF',
            ngram_counts=[0, 81, 245],
            ngram_indexes=[1, 1, 4, 0, 3, 1, 3, 1, 1, 3, 1, 4, 3, 0, 4, 0, 1, 0, 4, 3, 0, 2, 4, 3, 0, 4, 3, 0, 1, 4, 4, 0, 2,
 0, 3, 3, 4, 0, 3, 0, 2, 1, 0, 3, 0, 4, 4, 0, 3, 1, 0, 0, 0, 4, 0, 0, 3, 4, 3, 3, 4, 3, 3, 0, 3, 0,
 1, 3, 0, 0, 2, 0, 0, 4, 4, 2, 2, 2, 2, 4, 4, 2, 3, 1, 2, 2, 4, 1, 4, 3, 4, 4, 4, 0, 1, 2, 0, 2, 4,
 0, 3, 0, 0, 4, 0, 4, 1, 4, 3, 0, 3, 4, 3, 4, 4, 4, 0, 3, 3, 4, 2, 4, 2, 0, 1, 1, 2, 4, 1, 4, 0, 4,
 0, 1, 2, 0, 4, 0, 3, 0, 3, 0, 0, 1, 4, 1, 0, 3, 0, 4, 2, 3, 2, 3, 0, 0, 3, 4, 0, 3, 0, 0, 4, 1, 1,
 3, 0, 2, 2, 3, 3, 4, 0, 0, 1, 1, 0, 0, 1, 0, 3, 0, 0, 1, 1, 2, 3, 0, 1, 0, 4],
            pool_int64s=[0, 15, 32, 70, 96, 97, 102, 109, 124, 136, 148, 151, 188, 256, 269, 341, 448, 486, 505, 537, 545,
 558, 563, 567, 653, 672, 735, 748, 785, 786, 851, 855, 893, 939, 978, 996, 1011, 1025, 1033, 1057,
 1063, 1079, 1084, 1091, 1206, 1289, 1293, 1295, 1334, 1339, 1374, 1390, 1391, 1406, 1409, 1436,
 1480, 1507, 1542, 1547, 1549, 1559, 1560, 1569, 1577, 1695, 1701, 1732, 1778, 1811, 1838, 1852,
 1864, 1871, 1905, 1914, 1927, 1941, 2012, 2020, 2029, 0, 269, 0, 1547, 70, 109, 96, 1390, 96, 1391,
 97, 672, 97, 1507, 136, 855, 148, 1293, 148, 1811, 151, 786, 151, 1025, 185, 1941, 256, 32, 256,
 1409, 269, 1547, 341, 996, 486, 563, 505, 2012, 545, 185, 545, 1941, 558, 893, 558, 1569, 567,
 1084, 567, 1871, 653, 1011, 653, 2020, 735, 341, 735, 996, 748, 505, 748, 2012, 785, 136, 785, 855,
 851, 1436, 893, 1569, 939, 1091, 951, 102, 951, 1927, 978, 486, 978, 563, 1025, 786, 1033, 1057,
 1079, 188, 1084, 15, 1084, 1374, 1091, 448, 1289, 1507, 1293, 1811, 1295, 1549, 1295, 2029, 1339,
 851, 1339, 1436, 1374, 15, 1390, 1391, 1406, 1206, 1406, 1905, 1480, 1334, 1480, 1542, 1507, 672,
 1542, 1334, 1549, 2029, 1559, 1577, 1559, 1695, 1560, 537, 1577, 1695, 1701, 70, 1732, 188, 1732,
 1079, 1778, 1289, 1778, 1507, 1838, 124, 1838, 1862, 1852, 1033, 1852, 1057, 1862, 124, 1864, 537,
 1864, 1560, 1871, 1084, 1914, 785, 1914, 1063, 1927, 102, 2020, 1011, 0, 269, 1547, 97, 1507, 672,
 148, 1293, 1811, 151, 1025, 786, 256, 32, 1409, 545, 185, 1941, 558, 893, 1569, 567, 1871, 1084,
 653, 2020, 1011, 735, 341, 996, 748, 505, 2012, 785, 136, 855, 939, 1091, 448, 951, 1927, 102, 978,
 486, 563, 1084, 1374, 15, 1295, 1549, 2029, 1339, 851, 1436, 1406, 1206, 1905, 1480, 1542, 1334,
 1559, 1577, 1695, 1701, 70, 109, 1732, 1079, 188, 1778, 1289, 1507, 1838, 1862, 124, 1852, 1033,
 1057, 1864, 1560, 537, 1914, 785, 1063],
        ),
        # 3: Einsum inputs=16 outputs=1
        _node(
            'Einsum',
            ['input', 'c', 'U3', 'P', 'B', 'P', 'B', 'V3', 'P', 'B', 'P', 'B', 'U3', 'P', 'E', 'E'],
            ['output'],
            name='',
            domain='',
            equation='bihw,ql,lt,ta,ah,td,dh,ls,se,ew,sf,fw,lu,ur,ir,or->bohw',
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
            _tensor('like', TensorProto.FLOAT16, (1, 3), INIT_LIKE_0),
            _tensor('B', TensorProto.FLOAT, (2, 30), INIT_B_1),
            _tensor('U3', TensorProto.FLOAT, (5, 3), INIT_U3_2),
            _tensor('P', TensorProto.FLOAT, (3, 2), INIT_P_3),
            _tensor('V3', TensorProto.FLOAT, (5, 3), INIT_V3_4),
            _tensor('E', TensorProto.FLOAT, (10, 2), INIT_E_5),
        ],
        value_info=[
            _vi('rf', TensorProto.FLOAT16, [1, 3]),
            _vi('tok', TensorProto.INT32, [1, 3]),
            _vi('c', TensorProto.FLOAT, [1, 5]),
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
