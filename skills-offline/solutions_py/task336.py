from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task336'
TASK_NUM = 336
KAGGLE = {'score': 20.617973, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 20)]


# U: FLOAT[2, 10], 20 value(s)
INIT_U_0 = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

# Z: FLOAT[2, 30], 60 value(s)
INIT_Z_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0017540454864502, -0.8321459889411926, 1.79465913772583,
 -1.0016471147537231, 0.09006806463003159, -0.26419901847839355, -0.8319541215896606, -1.0014292001724243,
 -0.8524789214134216, -0.8316829800605774, -1.001425862312317, -1.0017061233520508, -0.8322222232818604,
 0.7171760201454163, -0.3805425465106964, 0.15079279243946075, -0.8319858908653259, -1.0021066665649414,
 -0.8319206237792969, -1.8201053142547607, -1.0, -0.7777777910232544, -0.5555555820465088, -0.3333333432674408,
 -0.1111111119389534, 0.1111111119389534, 0.3333333432674408, 0.5555555820465088, 0.7777777910232544, 1.0,
 0.7368216514587402, -0.9403932094573975, 1.4886798858642578, 0.7368450164794922, -2.259896993637085,
 -1.1876318454742432, -0.9404315948486328, 0.736057460308075, -0.14294461905956268, -0.94048011302948,
 0.736053466796875, 0.7370723485946655, -0.9392603635787964, -1.7211592197418213, 2.1685078144073486,
 1.9924687147140503, -0.9403991103172302, 0.738771915435791, -0.9404372572898865, -1.4455522298812866]


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
        # 0: Einsum inputs=64 outputs=1
        _node(
            'Einsum',
            ['Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'U', 'Z', 'input', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z',
             'Z', 'Z', 'Z', 'input', 'U', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'input', 'U', 'Z',
             'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'input', 'U', 'Z', 'Z', 'Z',
             'Z', 'Z', 'Z', 'Z', 'Z', 'input', 'U', 'U'],
            ['output'],
            name='output',
            domain='',
            equation='ur,ur,hr,hs,hs,hs,hs,ps,ps,qs,uc,qk,bcik,Wi,Wy,Wy,hy,hy,hy,hy,Vy,Qm,bdim,vd,QS,hS,hS,hS,hS,PS,PS,pj,Pj,Vl,belj,we,vR,vR,hR,wx,wx,hx,Ki,KN,KN,hN,hN,hN,hN,LN,Ln,bfnj,gf,gM,gM,hM,ht,ht,ht,ht,zt,baij,za,zo->boij',
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
            _tensor('U', TensorProto.FLOAT, (2, 10), INIT_U_0),
            _tensor('Z', TensorProto.FLOAT, (2, 30), INIT_Z_1),
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
