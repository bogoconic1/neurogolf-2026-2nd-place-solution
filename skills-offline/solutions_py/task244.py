from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task244'
TASK_NUM = 244
KAGGLE = {'score': 21.503492, 'date': '2026-07-12'}
MEMORY_BYTES = 31
PARAMS = 2
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task244_cost33'
OPSETS = [('', 16)]


# prob: FLOAT[1, 1], 1 value(s)
INIT_PROB_0 = [0.5799999833106995]

# batch_idx: INT64[1], 1 value(s)
INIT_BATCH_IDX_1 = [0]


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
            ['prob'],
            ['u'],
            name='',
            domain='',
            dtype=3,
            seed=18846406.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['prob'],
            ['v'],
            name='',
            domain='',
            dtype=3,
            seed=751522816.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['prob'],
            ['k'],
            name='',
            domain='',
            dtype=3,
            seed=46356912.0,
        ),
        # 3: Add inputs=2 outputs=1
        _node(
            'Add',
            ['v', 'v'],
            ['v2'],
            name='',
            domain='',
        ),
        # 4: Add inputs=2 outputs=1
        _node(
            'Add',
            ['u', 'v2'],
            ['d'],
            name='',
            domain='',
        ),
        # 5: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['d'],
            ['p'],
            name='',
            domain='',
            bias=3.0,
            lambd=-100.0,
        ),
        # 6: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['k'],
            ['k2'],
            name='',
            domain='',
            bias=2.0,
            lambd=-100.0,
        ),
        # 7: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['p', 'k2'],
            ['x1'],
            name='',
            domain='',
        ),
        # 8: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['v'],
            ['vm20'],
            name='',
            domain='',
            bias=-20.0,
            lambd=-100.0,
        ),
        # 9: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['p', 'vm20'],
            ['x2'],
            name='',
            domain='',
        ),
        # 10: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['k', 'x2'],
            ['y2'],
            name='',
            domain='',
        ),
        # 11: Concat inputs=4 outputs=1
        _node(
            'Concat',
            ['x1', 'v', 'x2', 'y2'],
            ['raw'],
            name='',
            domain='',
            axis=1,
        ),
        # 12: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['raw'],
            ['rois'],
            name='',
            domain='',
            to=1,
        ),
        # 13: RoiAlign inputs=3 outputs=1
        _node(
            'RoiAlign',
            ['input', 'rois', 'batch_idx'],
            ['output'],
            name='',
            domain='',
            coordinate_transformation_mode='half_pixel',
            mode='avg',
            output_height=30,
            output_width=30,
            sampling_ratio=1,
            spatial_scale=1.3359999656677246,
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
            _tensor('prob', TensorProto.FLOAT, (1, 1), INIT_PROB_0),
            _tensor('batch_idx', TensorProto.INT64, (1,), INIT_BATCH_IDX_1),
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
