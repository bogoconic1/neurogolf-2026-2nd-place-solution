from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task178'
TASK_NUM = 178
KAGGLE = {'score': 20.384879, 'date': '2026-07-13'}
MEMORY_BYTES = 54
PARAMS = 47
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'crt'
OPSETS = [('', 19)]


# source: FLOAT16[1], 1 value(s)
INIT_SOURCE_0 = [1.0]

# mods: INT64[1, 1, 1, 5], 5 value(s)
INIT_MODS_1 = [8112, 7207, 8161, 7291, 8228]

# vals: FLOAT16[1, 1, 1, 5], 5 value(s)
INIT_VALS_2 = [1.0, 1.0, 1.0, 1.0, 1.0]

# oshape: INT64[4], 4 value(s)
INIT_OSHAPE_3 = [1, 10, 30, 30]

# payload: INT64[32], 32 value(s)
INIT_PAYLOAD_4 = [6693068828902014480, 7099357657341001932, 7143333018782865972, 1158544749253364160, 5161352351160417420,
 2061971823537524352, 4986329073525637476, 5997932506366875696, 7116152506622319816, 877139781380114016,
 4899305116898744028, 674639315035726212, 1671077623856197296, 1706035390657459332, 3169115342096763000,
 6211323863371721388, 0, 161818950696561528, 2041998960424145484, 998399105903021604, 5728060449273981516,
 2138802638977173168, 1274042334321408492, 2848310668698375216, 1343683174511841132, 3277519690051565496,
 598884251326530480, 2461433420415810252, 6602878994641761300, 5852965799747529276, 4316097446908270944,
 5123869860632158476]


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
            ['source'],
            ['rf'],
            name='',
            domain='',
            dtype=10,
            high=3.9803617000579834,
            low=-29.133535385131836,
            seed=26998238.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['rf'],
            ['key'],
            name='',
            domain='',
            to=6,
        ),
        # 2: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['payload', 'key'],
            ['packed'],
            name='',
            domain='',
            axis=0,
        ),
        # 3: Mod inputs=2 outputs=1
        _node(
            'Mod',
            ['packed', 'mods'],
            ['idx'],
            name='',
            domain='',
        ),
        # 4: MaxUnpool inputs=3 outputs=1
        _node(
            'MaxUnpool',
            ['vals', 'idx', 'oshape'],
            ['output'],
            name='',
            domain='',
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.FLOAT16, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('source', TensorProto.FLOAT16, (1,), INIT_SOURCE_0),
            _tensor('mods', TensorProto.INT64, (1, 1, 1, 5), INIT_MODS_1),
            _tensor('vals', TensorProto.FLOAT16, (1, 1, 1, 5), INIT_VALS_2),
            _tensor('oshape', TensorProto.INT64, (4,), INIT_OSHAPE_3),
            _tensor('payload', TensorProto.INT64, (32,), INIT_PAYLOAD_4),
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
