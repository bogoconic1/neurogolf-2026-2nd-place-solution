from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task015'
TASK_NUM = 15
KAGGLE = {'score': 20.522663, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 88
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task015_m0_p88_noB'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task015_m0_p88_noB'
OPSETS = [('', 18)]


# X: FLOAT[30, 2], 60 value(s)
INIT_X_0 = [1.0708552598953247, 0.0011337161995470524, 1.0054852962493896, 0.3701920509338379, 0.8166967630386353,
 0.6929698586463928, 0.5296164155006409, 0.9288126826286316, 0.18269510567188263, 1.054742455482483,
 -0.18472425639629364, 1.0541975498199463, -0.5303322076797485, 0.9292110204696655, -0.814159095287323,
 0.6930692791938782, -1.0016162395477295, 0.3728766143321991, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# S: FLOAT[2, 2, 2], 8 value(s)
INIT_S_1 = [-0.9019189476966858, -3.992295980453491, 3.9865875244140625, -0.8821117281913757, 1.3331851959228516,
 3.5367300510406494, -3.5286736488342285, 1.3083889484405518]

# C: FLOAT[10, 2], 20 value(s)
INIT_C_2 = [-0.03406168892979622, -1.1424354314804077, -2.0290517807006836, -2.5382113456726074, 2.9248788356781006,
 -1.7521624565124512, -1.539382815361023, -0.5045446157455444, 0.28984132409095764, -0.8923744559288025,
 -1.4949398040771484, -0.48787668347358704, -1.2978603839874268, -0.026220323517918587, 1.016544222831726,
 -2.2165544033050537, 0.940801203250885, 0.548367977142334, -1.500549077987671, -0.49304676055908203]


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
        # 0: Einsum inputs=95 outputs=1
        _node(
            'Einsum',
            ['S', 'S', 'S', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X',
             'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'S', 'X', 'S', 'S', 'X', 'S', 'X', 'input', 'C', 'C', 'S',
             'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'X', 'S', 'S', 'X',
             'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X',
             'X', 'X', 'X', 'X', 'X', 'S', 'S', 'S', 'X', 'S', 'S', 'S', 'S', 'S', 'C', 'C'],
            ['output'],
            name='task015_m0_p88_noB',
            domain='',
            equation='QRR,QUV,QST,wU,wT,sS,sV,wP,sP,sO,wO,sE,wE,sI,wI,wK,sK,wJ,sJ,wH,sH,wL,sL,wF,sF,wG,sG,wX,qWX,sW,qqq,qqq,sZ,qYZ,wY,...crs,cm,cN,qmn,qnn,qnn,qnn,nqn,nnq,qNN,qNN,qNN,NqN,NNq,NNN,NNN,NNN,NNN,NNN,qMN,rA,qAB,qCD,rD,hC,hB,rb,hb,hg,rg,rf,hf,hd,rd,rj,hj,ra,ha,rl,hl,he,re,hp,rp,ri,hi,rz,rv,hx,tvx,tyz,tuu,hy,nnn,nnn,nnn,nnn,nnn,on,oM->...ohw',
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
            _tensor('X', TensorProto.FLOAT, (30, 2), INIT_X_0),
            _tensor('S', TensorProto.FLOAT, (2, 2, 2), INIT_S_1),
            _tensor('C', TensorProto.FLOAT, (10, 2), INIT_C_2),
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
