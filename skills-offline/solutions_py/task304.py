from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task304'
TASK_NUM = 304
KAGGLE = {'score': 20.087345, 'date': '2026-07-15'}
MEMORY_BYTES = 8
PARAMS = 128
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'gpt-5.6-sol/swarm-20plus'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task304_projective_power16'
OPSETS = [('', 18)]


# cw: FLOAT[10, 2], 20 value(s)
INIT_CW_0 = [0.9250797629356384, 0.007713819853961468, 0.7826376557350159, 1.6313867568969727, 1.1492048501968384,
 3.050726890563965, 0.8638826012611389, 2.777318000793457, 0.8929007053375244, 3.3181605339050293, 1.4605520963668823,
 4.318551540374756, 1.2391934394836426, 5.203863143920898, 1.5291849374771118, 6.510898113250732, 1.3180129528045654,
 6.724230766296387, 1.5027226209640503, 6.991851329803467]

# F: FLOAT[2, 2, 2], 8 value(s)
INIT_F_1 = [0.733889639377594, 0.07025404274463654, 0.06419461220502853, -0.032451849430799484, 0.441957950592041,
 1.6299264430999756, -1.8929779529571533, 0.03676623851060867]

# G: FLOAT[2, 2], 4 value(s)
INIT_G_2 = [0.038601119071245193, -0.40323606133461, -0.014965290203690529, 2.311546802520752]

# Crd: FLOAT[30, 3], 90 value(s)
INIT_CRD_3 = [1.0, -1.0, 0.0, 1.0, -0.5, 0.5, 1.0, 0.0, 1.0, 1.0, -0.5, -0.5, 1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, -1.0, 1.0, 0.5,
 -0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# A: FLOAT[3, 2], 6 value(s)
INIT_A_4 = [1.0, 0.0, 0.0, 1.0, 0.0, -1.0]


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
        # 0: Einsum inputs=17 outputs=1
        _node(
            'Einsum',
            ['input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input',
             'input', 'input', 'input', 'input', 'input', 'cw'],
            ['mvec'],
            name='',
            domain='',
            equation='...xab,...xcd,...xef,...xgh,...xij,...xkl,...xmn,...xpq,...xrs,...xtu,...xvw,...xyz,...xAB,...xCD,...xEF,...xGH,xo->...o',
        ),
        # 1: Einsum inputs=77 outputs=1
        _node(
            'Einsum',
            ['mvec', 'F', 'mvec', 'F', 'cw', 'cw', 'A', 'A', 'Crd', 'Crd', 'input', 'Crd', 'Crd', 'A', 'A', 'A',
             'A', 'A', 'A', 'F', 'G', 'F', 'G', 'F', 'G', 'F', 'G', 'A', 'A', 'Crd', 'Crd', 'Crd', 'A', 'A',
             'F', 'G', 'Crd', 'A', 'A', 'F', 'G', 'A', 'A', 'A', 'A', 'Crd', 'Crd', 'input', 'cw', 'cw', 'Crd',
             'Crd', 'A', 'A', 'A', 'A', 'F', 'G', 'F', 'F', 'F', 'G', 'F', 'G', 'A', 'A', 'A', 'Crd', 'Crd',
             'A', 'Crd', 'A', 'A', 'Crd', 'cw', 'cw', 'cw'],
            ['output'],
            name='',
            domain='',
            equation='...o,qoc,...p,qdp,gc,gd,Zw,Zw,aZ,aJ,...gab,bU,bV,Ju,Ju,Uy,Uy,VA,VA,xxy,xy,ttu,tu,wwv,wv,AAz,Az,Pv,It,hP,hI,hO,OB,OB,BBC,BC,hY,YD,YD,EED,ED,XC,XC,TE,TE,rX,rT,...Qrs,Qf,Qe,sS,sM,SG,SG,Mn,Mn,Kqj,Kj,jie,jfl,nnH,nH,FFG,FG,WF,WF,Rx,mR,mW,Lz,mL,NH,NH,mN,ki,kl,kK->...khm',
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
            _tensor('cw', TensorProto.FLOAT, (10, 2), INIT_CW_0),
            _tensor('F', TensorProto.FLOAT, (2, 2, 2), INIT_F_1),
            _tensor('G', TensorProto.FLOAT, (2, 2), INIT_G_2),
            _tensor('Crd', TensorProto.FLOAT, (30, 3), INIT_CRD_3),
            _tensor('A', TensorProto.FLOAT, (3, 2), INIT_A_4),
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
