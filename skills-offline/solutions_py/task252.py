from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task252'
TASK_NUM = 252
KAGGLE = {'score': 21.087977, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 50
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task252_p50_synthD'
OPSETS = [('', 18)]


# E: FLOAT[10, 2], 20 value(s)
INIT_E_0 = [-0.5067062973976135, 0.04493017867207527, 0.26583370566368103, -0.45808055996894836, 0.48833638429641724,
 -0.4639740586280823, -0.18568600714206696, 0.4374984800815582, -0.005400166381150484, 0.5942132472991943,
 0.17435681819915771, 0.4491932988166809, -0.09141848981380463, -0.4365101158618927, 0.06270841509103775,
 -0.46166524291038513, 0.3195869028568268, -0.4343436658382416, 0.12451963126659393, -0.4480094313621521]

# Q: FLOAT[30], 30 value(s)
INIT_Q_1 = [1.0, 1.3452465257518244e-43, 1.0, 1.3452465257518244e-43, 1.0, 1.3452465257518244e-43, 1.0, 1.3452465257518244e-43,
 1.0, 1.3452465257518244e-43, 1.0, 1.3452465257518244e-43, 1.0, 1.3452465257518244e-43, 1.0, 1.3452465257518244e-43,
 1.0, 1.3452465257518244e-43, 1.0, 1.3452465257518244e-43, 1.0, 1.3452465257518244e-43, 1.0, 1.3452465257518244e-43,
 1.0, 1.3452465257518244e-43, 1.0, 1.3452465257518244e-43, 1.0, 1.3452465257518244e-43]


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
        # 0: Einsum inputs=22 outputs=1
        _node(
            'Einsum',
            ['E', 'E', 'E', 'E', 'E', 'E', 'input', 'Q', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
             'E', 'E', 'E'],
            ['output'],
            name='',
            domain='',
            equation='ug,tg,ta,ta,ta,ta,nchw,w,ca,ce,za,za,zb,zb,zb,ob,yd,yd,ye,ye,ye,od->nohw',
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
            _tensor('E', TensorProto.FLOAT, (10, 2), INIT_E_0),
            _tensor('Q', TensorProto.FLOAT, (30,), INIT_Q_1),
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
