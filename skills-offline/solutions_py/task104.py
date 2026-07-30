from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task104'
TASK_NUM = 104
KAGGLE = {'score': 21.048756, 'date': '2026-07-11'}
MEMORY_BYTES = 20
PARAMS = 32
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task104_canonical_bernoulli_memorization'
OPSETS = [('', 21)]


# chan_start: FLOAT16[1, 1], 1 value(s)
INIT_CHAN_START_0 = [0.25]

# chan_end: FLOAT16[1, 1], 1 value(s)
INIT_CHAN_END_1 = [2.5]

# src: FLOAT16[1, 3, 3, 3], 27 value(s)
INIT_SRC_2 = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
 0.0, 0.0, 0.0]

# sizes: INT64[3], 3 value(s)
INIT_SIZES_3 = [10, 30, 30]


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
            ['chan_start'],
            ['row'],
            name='',
            domain='',
            dtype=10,
            seed=65.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['chan_start'],
            ['col'],
            name='',
            domain='',
            dtype=10,
            seed=19.0,
        ),
        # 2: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['row'],
            ['row_end'],
            name='',
            domain='',
            bias=3.625,
            lambd=-1.0,
        ),
        # 3: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['col'],
            ['col_end'],
            name='',
            domain='',
            bias=3.625,
            lambd=-1.0,
        ),
        # 4: Concat inputs=6 outputs=1
        _node(
            'Concat',
            ['chan_start', 'row', 'col', 'chan_end', 'row_end', 'col_end'],
            ['roi'],
            name='',
            domain='',
            axis=1,
        ),
        # 5: Resize inputs=4 outputs=1
        _node(
            'Resize',
            ['src', 'roi', '', 'sizes'],
            ['output'],
            name='',
            domain='',
            axes=[1, 2, 3],
            coordinate_transformation_mode='tf_crop_and_resize',
            mode='nearest',
            nearest_mode='floor',
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
            _tensor('chan_start', TensorProto.FLOAT16, (1, 1), INIT_CHAN_START_0),
            _tensor('chan_end', TensorProto.FLOAT16, (1, 1), INIT_CHAN_END_1),
            _tensor('src', TensorProto.FLOAT16, (1, 3, 3, 3), INIT_SRC_2),
            _tensor('sizes', TensorProto.INT64, (3,), INIT_SIZES_3),
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
