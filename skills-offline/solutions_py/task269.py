from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task269'
TASK_NUM = 269
KAGGLE = {'score': 19.412751, 'date': '2026-07-15'}
MEMORY_BYTES = 241
PARAMS = 26
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task269_pool_power5_einsum'
OPSETS = [('', 18)]


# code_w: FLOAT[10, 1, 1, 1], 10 value(s)
INIT_CODE_W_0 = [-41.0, 4194380.0, 4194319.0, 4194232.0, 4194349.0, 4194370.0, 4194368.0, 4194336.0, 4194240.0, 4194271.0]

# roi_num: FLOAT[4], 4 value(s)
INIT_ROI_NUM_1 = [0.0, 0.0, 1.298074214633707e+34, 1.298074214633707e+34]

# mults: INT8[1, 10, 1, 1], 10 value(s)
INIT_MULTS_2 = [-52, -74, -98, -122, -106, -92, -35, 26, -105, -24]

# sizes: INT64[2], 2 value(s)
INIT_SIZES_3 = [30, 30]


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
        # 0: ConvTranspose inputs=2 outputs=1
        _node(
            'ConvTranspose',
            ['input', 'code_w'],
            ['code_f'],
            name='',
            domain='',
            kernel_shape=[1, 1],
            pads=[0, 0, 27, 27],
        ),
        # 1: Einsum inputs=5 outputs=1
        _node(
            'Einsum',
            ['code_f', 'code_f', 'code_f', 'code_f', 'code_f'],
            ['sum_code'],
            name='',
            domain='',
            equation='nchw,nchw,nchw,nchw,nchw->',
        ),
        # 2: Div inputs=2 outputs=1
        _node(
            'Div',
            ['roi_num', 'sum_code'],
            ['roi'],
            name='',
            domain='',
        ),
        # 3: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['code_f'],
            ['x'],
            name='',
            domain='',
            to=3,
        ),
        # 4: MaxPool inputs=1 outputs=1
        _node(
            'MaxPool',
            ['x'],
            ['x4'],
            name='',
            domain='',
            dilations=[4, 4],
            kernel_shape=[3, 3],
            pads=[0, 0, -5, -5],
            strides=[-3, -3],
        ),
        # 5: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['x4', 'mults'],
            ['src'],
            name='',
            domain='',
        ),
        # 6: Resize inputs=4 outputs=1
        _node(
            'Resize',
            ['src', 'roi', '', 'sizes'],
            ['output'],
            name='',
            domain='',
            axes=[2, 3],
            coordinate_transformation_mode='tf_crop_and_resize',
            extrapolation_value=0.0,
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
            _vi('output', TensorProto.INT8, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('code_w', TensorProto.FLOAT, (10, 1, 1, 1), INIT_CODE_W_0),
            _tensor('roi_num', TensorProto.FLOAT, (4,), INIT_ROI_NUM_1),
            _tensor('mults', TensorProto.INT8, (1, 10, 1, 1), INIT_MULTS_2),
            _tensor('sizes', TensorProto.INT64, (2,), INIT_SIZES_3),
        ],
        value_info=[
            _vi('code_f', TensorProto.FLOAT, [1, 1, 3, 3]),
            _vi('roi', TensorProto.FLOAT, [4]),
            _vi('x', TensorProto.INT8, [1, 1, 3, 3]),
            _vi('x4', TensorProto.INT8, [1, 1, 4, 4]),
            _vi('src', TensorProto.INT8, [1, 10, 4, 4]),
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
