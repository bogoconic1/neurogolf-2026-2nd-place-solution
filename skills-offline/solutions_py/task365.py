from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task365'
TASK_NUM = 365
KAGGLE = {'score': 20.094725, 'date': '2026-07-14'}
MEMORY_BYTES = 22
PARAMS = 113
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task365_seeded_rng_inttable_direct_rank2'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1, 1], 1 value(s)
INIT_LIKE_0 = [0.0]

# CASES: INT64[31, 2], 62 value(s)
INIT_CASES_1 = [-1189513251647389697, -219550482069192704, 2263970307814211583, -1233986297914195968, 9113315320851857407,
 -8070591269746769920, -198158383618981889, 4539346949362417664, -182958734861926401, 6916403127666081792,
 -3914050288660774913, -46161896391311360, -254462175045812225, -6917564212147388416, -2306969055285937177,
 9221110891155554304, 7943316201709436927, -577588851336282112, -4667981013770567681, 8068761682350964736,
 -2514415966958387201, -5767000060479930368, 9079256848772628479, -6955809624474779648, 3204311134806999039,
 -2307813334319104000, -1894896140289671169, -4656757199242002432, 2268688312287887359, -218987532149325824,
 -36028797033644033, -7133701809889083392, -2451084097339195393, 4611114272379895808, -3686477770114203649,
 -596164000874496000, 6789739388214198271, -4758334481310416896, 8034140260245962751, -164381386500734976,
 4339781190906281983, -875950127573893120, 4014097050512515071, -72211525718245376, -218477358491893761,
 4575648425281847296, -8796176011009, -108103983496691712, -8796176011009, -108103983496691712, 9219967667400472543,
 -144678275544776704, -4614500768194494465, -253327479207362560, 7995578188304482303, -4774097080022990848,
 4743697782282387455, -4683954718966284288, -38280669921774125, 9223367363826548736, -37154696927903745,
 -7135953609702768640]

# P: INT64[30], 30 value(s)
INIT_P_2 = [1, 2, 4, 8, 16, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# D: INT64[2, 10], 20 value(s)
INIT_D_3 = [0, 1, 0, 0, 0, 0, 0, 0, 268435456, 0, 0, 0, 1, 0, 0, 0, 0, 0, 68719476736, 0]


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
            ['LIKE'],
            ['rng_f'],
            name='',
            domain='',
            dtype=10,
            high=4.375,
            low=-27.3125,
            seed=891039552.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['rng_f'],
            ['case_i'],
            name='',
            domain='',
            to=6,
        ),
        # 2: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['CASES', 'case_i'],
            ['packed'],
            name='',
            domain='',
            axis=0,
        ),
        # 3: Einsum inputs=9 outputs=1
        _node(
            'Einsum',
            ['packed', 'D', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['output'],
            name='',
            domain='',
            equation='bqk,kc,r,r,r,r,r,r,w->bcrw',
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.INT64, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('LIKE', TensorProto.FLOAT16, (1, 1), INIT_LIKE_0),
            _tensor('CASES', TensorProto.INT64, (31, 2), INIT_CASES_1),
            _tensor('P', TensorProto.INT64, (30,), INIT_P_2),
            _tensor('D', TensorProto.INT64, (2, 10), INIT_D_3),
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
