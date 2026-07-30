from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task001'
TASK_NUM = 1
KAGGLE = {'score': 20.695935, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 74
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task001_p74_exact_label21'
OPSETS = [('', 18)]


# F: FLOAT[30, 2], 60 value(s)
INIT_F_0 = [0.8852664828300476, 0.8852664828300476, 1.756085991859436, 0.0, 1.5008139610290527, -0.7504069805145264,
 1.2989243268966675, -1.2989243268966675, 0.8037590384483337, -1.6075180768966675, 0.0, 1.9936991930007935,
 0.7398459315299988, 1.4796918630599976, 0.609521210193634, 1.8285636901855469, 1.8264758586883545, 0.9132379293441772,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# M: FLOAT[2, 2], 4 value(s)
INIT_M_1 = [1.0, 1.0, 2.0, -1.0]

# sign: FLOAT[10], 10 value(s)
INIT_SIGN_2 = [-1.0, 3.0491424695355818e-05, 3.0491424695355818e-05, 3.201115760020912e-05, 3.0491424695355818e-05,
 3.0491424695355818e-05, 3.360662958584726e-05, 3.0491424695355818e-05, 0.0011078943498432636, 3.201115760020912e-05]


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
        # 0: Einsum inputs=116 outputs=1
        _node(
            'Einsum',
            ['input', 'sign', 'F', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'M',
             'M', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'M',
             'M', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'M', 'M', 'M', 'F',
             'F', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'M', 'M', 'M', 'F',
             'F', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'M', 'M', 'M', 'F',
             'F', 'M', 'M', 'M', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'M', 'M', 'M', 'F', 'input', 'sign'],
            ['output'],
            name='output',
            domain='',
            equation='ncxy,c,xa,ba,ad,dd,rd,xe,ef,ef,ef,fe,ff,rf,xg,bg,bg,gb,gb,gh,hg,rh,xk,km,ll,lm,rl,xp,bt,qt,tb,tp,tp,rq,xu,bv,uv,uw,vb,vb,vw,rv,xz,bB,zb,zz,zB,AB,BB,rA,xC,Cb,CE,CE,Eb,EC,ED,rD,yF,bF,FG,GG,sG,yH,HI,HI,HI,IH,II,sI,yJ,bJ,bJ,Jb,Jb,JK,KJ,sK,yL,LN,MM,MN,sM,yO,bQ,PQ,Qb,QO,QO,sP,yR,bS,RS,RT,Sb,Sb,ST,sS,yU,bW,Ub,UU,UW,VW,WW,sV,yX,Xb,XZ,XZ,Zb,ZX,ZY,sY,noij,o->nors',
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
            _tensor('F', TensorProto.FLOAT, (30, 2), INIT_F_0),
            _tensor('M', TensorProto.FLOAT, (2, 2), INIT_M_1),
            _tensor('sign', TensorProto.FLOAT, (10,), INIT_SIGN_2),
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
