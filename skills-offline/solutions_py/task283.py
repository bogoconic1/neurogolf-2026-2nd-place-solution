from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task283'
TASK_NUM = 283
KAGGLE = {'score': 20.522663, 'date': '2026-07-11'}
MEMORY_BYTES = 0
PARAMS = 88
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'p88_B_reorder_a1_b4'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'p88_B_reorder_a1_b4'
OPSETS = [('', 18)]


# V: FLOAT[30, 2], 60 value(s)
INIT_V_0 = [1.0, 0.0, 0.8412535190582275, 0.5406408309936523, 0.4154150187969208, 0.9096319675445557, -0.1423148363828659,
 0.9898214340209961, -0.6548607349395752, 0.7557495832443237, -0.9594929814338684, 0.28173255920410156,
 -0.9594929814338684, -0.28173255920410156, -0.6548607349395752, -0.7557495832443237, -0.1423148363828659,
 -0.9898214340209961, 0.4154150187969208, -0.9096319675445557, -1.7720807790756226, 3.9824934005737305,
 2.478915214538574, -4.229893207550049, -2.938469171524048, 4.231911659240723, 3.0400896072387695, -3.967235565185547,
 -2.409200429916382, 2.3947596549987793, -3.5302295684814453, 2.294372081756592, 3.816448450088501, -1.6058307886123657,
 -4.088006496429443, 0.40463781356811523, 4.199187755584717, 0.528469979763031, -4.206927299499512, -1.7681611776351929,
 4.094428062438965, 2.5724058151245117, -3.752246379852295, -3.4325344562530518, 3.445997714996338, 4.444576263427734,
 -3.2841312885284424, -4.724037170410156, 2.835073947906494, 4.648081302642822, -1.957678198814392, -4.217349052429199,
 -0.7048432230949402, -0.8372925519943237, -1.7118799686431885, 0.4054999351501465, 0.282306432723999,
 1.2871770858764648, -0.14609229564666748, 0.23643329739570618]

# Rmode: FLOAT[2, 2, 2], 8 value(s)
INIT_RMODE_1 = [-0.28173255920410156, -0.43838444352149963, 0.43838444352149963, -0.28173255920410156, -0.0, -0.85349041223526,
 0.85349041223526, -0.0]

# E: FLOAT[2, 10], 20 value(s)
INIT_E_2 = [-0.009658584371209145, 1.211150884628296, 0.2572806179523468, -0.0, -0.8923712968826294, 1.5917811393737793, -0.0,
 -0.0, -0.0, -0.0, -1.1927913427352905, 0.04106327146291733, 1.1735990047454834, -0.0, -0.9439203143119812,
 0.6663153171539307, -0.0, -0.0, -0.0, -0.0]


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
        # 0: Einsum inputs=85 outputs=1
        _node(
            'Einsum',
            ['V', 'V', 'V', 'V', 'V', 'Rmode', 'Rmode', 'V', 'V', 'V', 'Rmode', 'V', 'Rmode', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'Rmode', 'V', 'V', 'Rmode',
             'V', 'input', 'V', 'Rmode', 'Rmode', 'V', 'E', 'input', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'Rmode', 'Rmode', 'V', 'V', 'V', 'Rmode',
             'Rmode', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'E', 'E'],
            ['output'],
            name='p88_B_reorder_a1_b4',
            domain='',
            equation='XY,XE,XE,XE,XE,YJK,YLM,wM,wJ,wN,nNO,wQ,nPQ,yP,yL,yK,yO,yU,wU,yS,wS,wV,yV,wW,yW,wR,yR,yT,wT,wF,lFG,yG,wI,lHI,yH,...cxy,xi,Eij,Efg,xg,pc,...chw,hj,hf,hB,xB,xA,hA,xC,hC,xz,hz,xv,hv,xu,hu,xr,xb,xd,xs,Zst,Zmr,ht,hm,he,kde,kab,ha,Dk,Dl,Dl,Dp,Dp,Dp,Dp,Dq,Dq,Dq,Dq,Dq,Dq,Dq,Dq,po,qo->...ohw',
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
            _tensor('V', TensorProto.FLOAT, (30, 2), INIT_V_0),
            _tensor('Rmode', TensorProto.FLOAT, (2, 2, 2), INIT_RMODE_1),
            _tensor('E', TensorProto.FLOAT, (2, 10), INIT_E_2),
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
