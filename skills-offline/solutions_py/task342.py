from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task342'
TASK_NUM = 342
KAGGLE = {'score': 20.617973, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task342_m0_p80'
OPSETS = [('', 18)]


# P: FLOAT[2, 30], 60 value(s)
INIT_P_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6961426734924316, 0.6574253439903259, 0.7266449332237244,
 0.850029706954956, 0.657489001750946, 1.719398856163025, -1.4361944198608398, -1.7782011032104492, 0.3087264597415924,
 -0.4021368622779846, -1.3922855854034424, 1.8898876905441284, 0.6575992703437805, -0.7432225942611694,
 -1.2118474245071411, -1.7782045602798462, 1.1632709503173828, 0.6952839493751526, 0.8536221981048584,
 -1.2706819772720337, -2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 1.0808393955230713,
 1.0198254585266113, -0.012264659628272057, -0.012576219625771046, 1.0198938846588135, -1.6854966878890991,
 -1.8136085271835327, 0.3280920386314392, 0.44623059034347534, -2.128535509109497, 2.0391933917999268,
 0.4187653958797455, 1.0200210809707642, 1.9242498874664307, -1.878590703010559, 0.3280875086784363, 0.6077897548675537,
 1.0794477462768555, -0.012772929854691029, 2.0037107467651367]

# E: FLOAT[2, 10], 20 value(s)
INIT_E_1 = [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


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
        # 0: Einsum inputs=44 outputs=1
        _node(
            'Einsum',
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'input',
             'P', 'input', 'E', 'E', 'input', 'P', 'P', 'P', 'P', 'P', 'P', 'input', 'P', 'P', 'P', 'P', 'P',
             'P', 'E', 'P', 'input', 'P', 'P', 'input', 'P'],
            ['output'],
            name='',
            domain='',
            equation='mZ,mZ,mZ,mZ,mZ,pZ,qZ,pA,jA,jA,jA,tA,pC,iC,iC,iC,sC,th,behn,sH,beHW,me,me,befg,vg,vB,qB,lB,lB,lB,beDE,uE,qF,uF,kF,kF,kF,ma,jr,barc,lc,ix,bexy,ky->baxy',
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
            _tensor('P', TensorProto.FLOAT, (2, 30), INIT_P_0),
            _tensor('E', TensorProto.FLOAT, (2, 10), INIT_E_1),
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
