from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task049'
TASK_NUM = 49
KAGGLE = {'score': 20.956949, 'date': '2026-07-12'}
MEMORY_BYTES = 54
PARAMS = 3
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task049_colorsum'
OPSETS = [('', 16)]


# zc: FLOAT16[1, 1, 1, 1], 1 value(s)
INIT_ZC_0 = [0.0]

# pf: FLOAT16[1, 1], 1 value(s)
INIT_PF_1 = [0.150634765625]

# batch: INT64[1], 1 value(s)
INIT_BATCH_2 = [0]


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
            ['zc'],
            ['ch1'],
            name='',
            domain='',
            dtype=10,
            high=0.4096347391605377,
            low=-2.9056615829467773,
            seed=-32494680.0,
        ),
        # 1: RandomUniformLike inputs=1 outputs=1
        _node(
            'RandomUniformLike',
            ['zc'],
            ['ch3'],
            name='',
            domain='',
            dtype=10,
            high=0.6227551698684692,
            low=-3.3157336711883545,
            seed=15738148.0,
        ),
        # 2: RandomUniformLike inputs=1 outputs=1
        _node(
            'RandomUniformLike',
            ['zc'],
            ['ch4'],
            name='',
            domain='',
            dtype=10,
            high=0.7287501096725464,
            low=-2.074554681777954,
            seed=3744151.0,
        ),
        # 3: RandomUniformLike inputs=1 outputs=1
        _node(
            'RandomUniformLike',
            ['zc'],
            ['ch5'],
            name='',
            domain='',
            dtype=10,
            high=0.06681820005178452,
            low=-0.9989819526672363,
            seed=136878768.0,
        ),
        # 4: RandomUniformLike inputs=1 outputs=1
        _node(
            'RandomUniformLike',
            ['zc'],
            ['ch6'],
            name='',
            domain='',
            dtype=10,
            high=0.690363883972168,
            low=-2.876406669616699,
            seed=248340208.0,
        ),
        # 5: RandomUniformLike inputs=1 outputs=1
        _node(
            'RandomUniformLike',
            ['zc'],
            ['ch7'],
            name='',
            domain='',
            dtype=10,
            high=0.39464235305786133,
            low=-2.385589838027954,
            seed=779828800.0,
        ),
        # 6: RandomUniformLike inputs=1 outputs=1
        _node(
            'RandomUniformLike',
            ['zc'],
            ['ch8'],
            name='',
            domain='',
            dtype=10,
            high=0.8716411590576172,
            low=-3.1415324211120605,
            seed=2156248.0,
        ),
        # 7: RandomUniformLike inputs=1 outputs=1
        _node(
            'RandomUniformLike',
            ['zc'],
            ['ch9'],
            name='',
            domain='',
            dtype=10,
            high=0.885966420173645,
            low=-4.780036926269531,
            seed=12818912.0,
        ),
        # 8: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['zc', 'ch1', 'zc', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9'],
            ['sel'],
            name='',
            domain='',
            axis=1,
        ),
        # 9: ReduceL1 inputs=1 outputs=1
        _node(
            'ReduceL1',
            ['sel'],
            ['l1'],
            name='',
            domain='',
            axes=[1, 2],
            keepdims=0,
        ),
        # 10: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pf'],
            ['qw0'],
            name='',
            domain='',
            dtype=10,
            seed=99681728.0,
        ),
        # 11: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pf'],
            ['qw1'],
            name='',
            domain='',
            dtype=10,
            seed=-848803968.0,
        ),
        # 12: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pf'],
            ['qw2'],
            name='',
            domain='',
            dtype=10,
            seed=-129892376.0,
        ),
        # 13: Sum inputs=12 outputs=1
        _node(
            'Sum',
            ['qw0', 'qw0', 'qw1', 'qw1', 'qw1', 'qw1', 'qw2', 'qw2', 'qw2', 'qw2', 'qw2', 'qw2'],
            ['wx'],
            name='',
            domain='',
        ),
        # 14: Concat inputs=4 outputs=1
        _node(
            'Concat',
            ['pf', 'pf', 'wx', 'l1'],
            ['rois'],
            name='',
            domain='',
            axis=1,
        ),
        # 15: RoiAlign inputs=3 outputs=1
        _node(
            'RoiAlign',
            ['sel', 'rois', 'batch'],
            ['output'],
            name='',
            domain='',
            coordinate_transformation_mode='output_half_pixel',
            mode='avg',
            output_height=30,
            output_width=30,
            sampling_ratio=0,
            spatial_scale=0.8986499905586243,
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
            _tensor('zc', TensorProto.FLOAT16, (1, 1, 1, 1), INIT_ZC_0),
            _tensor('pf', TensorProto.FLOAT16, (1, 1), INIT_PF_1),
            _tensor('batch', TensorProto.INT64, (1,), INIT_BATCH_2),
        ],
        value_info=[
            _vi('ch1', TensorProto.FLOAT16, [1, 1, 1, 1]),
            _vi('ch3', TensorProto.FLOAT16, [1, 1, 1, 1]),
            _vi('ch4', TensorProto.FLOAT16, [1, 1, 1, 1]),
            _vi('ch5', TensorProto.FLOAT16, [1, 1, 1, 1]),
            _vi('ch6', TensorProto.FLOAT16, [1, 1, 1, 1]),
            _vi('ch7', TensorProto.FLOAT16, [1, 1, 1, 1]),
            _vi('ch8', TensorProto.FLOAT16, [1, 1, 1, 1]),
            _vi('ch9', TensorProto.FLOAT16, [1, 1, 1, 1]),
            _vi('sel', TensorProto.FLOAT16, [1, 10, 1, 1]),
            _vi('qw0', TensorProto.FLOAT16, [1, 1]),
            _vi('qw1', TensorProto.FLOAT16, [1, 1]),
            _vi('qw2', TensorProto.FLOAT16, [1, 1]),
            _vi('wx', TensorProto.FLOAT16, [1, 1]),
            _vi('rois', TensorProto.FLOAT16, [1, 4]),
            _vi('l1', TensorProto.FLOAT16, [1, 1]),
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
