from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task157'
TASK_NUM = 157
KAGGLE = {'score': 19.758253, 'date': '2026-07-14'}
MEMORY_BYTES = 22
PARAMS = 167
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task157_int_payload_replay'
OPSETS = [('', 20)]


# like: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# cases: INT64[28, 2], 56 value(s)
INIT_CASES_1 = [-8144443266750742529, -17180559409, 9161164177216110591, -5506785, -1384898666858283009, -36865, -5492280964103012353,
 -5100429313, -8881105612000198657, -8254638081, -1792997525792227329, -785, -3975662797101989889, -3221471601,
 -7965813309167370241, -30787267425377, -688508683755192321, -34362884209, -8326070527391170561, -238525473,
 8967511248665051135, -8590204129, -4450682331748892673, -37162708513, -8109857132066635777, -100699137,
 -5165628772593958913, -274886427905, -357473220422533121, -2061643032497, -8342926624760528897, -246298238492161,
 3538837805322993663, -4353, -4293637075908952065, -240737457, -4144437557087698945, -123149060525089,
 -8345346168816926721, -83889665, -4865576447420399617, -140741784503345, 1427068906333274111, -4653185,
 8712213479148224511, -68793145537, 634444185188958207, -20401666449, 4072942913003192319, -14242546594769,
 3576983419923464191, -7368865, -4888665641847881729, -131265, 1204749716781268991, -1023441425]

# A: INT64[2, 10], 20 value(s)
INIT_A_2 = [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

# R: INT64[2, 30], 60 value(s)
INIT_R_3 = [1073741824, 32768, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 35184372088832, 1073741824, 32768, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# W: INT64[30], 30 value(s)
INIT_W_4 = [16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


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
            ['rng_f'],
            name='',
            domain='',
            dtype=10,
            high=14.686503410339355,
            low=-17.553634643554688,
            seed=7649488.0,
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
            ['cases', 'case_i'],
            ['packed'],
            name='',
            domain='',
            axis=0,
        ),
        # 3: Einsum inputs=4 outputs=1
        _node(
            'Einsum',
            ['packed', 'A', 'R', 'W'],
            ['output'],
            name='',
            domain='',
            equation='bk,kc,kr,w->bcrw',
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
            _tensor('like', TensorProto.FLOAT16, (1,), INIT_LIKE_0),
            _tensor('cases', TensorProto.INT64, (28, 2), INIT_CASES_1),
            _tensor('A', TensorProto.INT64, (2, 10), INIT_A_2),
            _tensor('R', TensorProto.INT64, (2, 30), INIT_R_3),
            _tensor('W', TensorProto.INT64, (30,), INIT_W_4),
        ],
        value_info=[
            _vi('rng_f', TensorProto.FLOAT16, [1]),
            _vi('case_i', TensorProto.INT32, [1]),
            _vi('packed', TensorProto.INT64, [1, 2]),
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
