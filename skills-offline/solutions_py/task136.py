from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task136'
TASK_NUM = 136
KAGGLE = {'score': 20.179718, 'date': '2026-07-13'}
MEMORY_BYTES = 32
PARAMS = 92
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task136_A1_factored_into_C_Ptail'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task136_A1_factored_into_C_Ptail'
OPSETS = [('', 12)]


# P: FLOAT[2, 30], 60 value(s)
INIT_P_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.243544340133667, -1.4370431900024414, -2.1090610027313232,
 -3.689042806625366, 2.8260860443115234, -2.167187213897705, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, -12.58016586303711, 8.722604751586914,
 4.267212390899658, -7.463951587677002, 17.153854370117188, -21.924087524414062, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Q: FLOAT[2, 10], 20 value(s)
INIT_Q_1 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# U: FLOAT[2, 2, 2], 8 value(s)
INIT_U_2 = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0]

# C: FLOAT[2, 2], 4 value(s)
INIT_C_3 = [-1.0, -3.0, -1.0, 1.0]


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
        # 0: Einsum inputs=7 outputs=1
        _node(
            'Einsum',
            ['C', 'C', 'C', 'Q', 'Q', 'input', 'P'],
            ['RR'],
            name='extract_red_row',
            domain='',
            equation='tt,tp,tq,pc,qc,ncRC,aR->na',
        ),
        # 1: Einsum inputs=9 outputs=1
        _node(
            'Einsum',
            ['C', 'C', 'C', 'Q', 'Q', 'P', 'input', 'P', 'U'],
            ['RD'],
            name='extract_red_diag',
            domain='',
            equation='tt,tp,tq,pc,qc,jR,ncRC,kC,ajk->na',
        ),
        # 2: Einsum inputs=9 outputs=1
        _node(
            'Einsum',
            ['C', 'C', 'C', 'C', 'C', 'Q', 'Q', 'input', 'P'],
            ['BR'],
            name='extract_blue_row',
            domain='',
            equation='tt,tp,tq,pp,qq,pc,qc,ncRC,aR->na',
        ),
        # 3: Einsum inputs=11 outputs=1
        _node(
            'Einsum',
            ['C', 'C', 'C', 'C', 'C', 'Q', 'Q', 'P', 'input', 'P', 'U'],
            ['BD'],
            name='extract_blue_diag',
            domain='',
            equation='tt,tp,tq,pp,qq,pc,qc,jR,ncRC,kC,ajk->na',
        ),
        # 4: Einsum inputs=85 outputs=1
        _node(
            'Einsum',
            ['RR', 'P', 'U', 'P', 'P', 'U', 'RD', 'U', 'C', 'P', 'P', 'P', 'P', 'P', 'C', 'C', 'C', 'Q', 'Q',
             'C', 'C', 'BR', 'P', 'U', 'P', 'P', 'U', 'BD', 'U', 'C', 'P', 'P', 'P', 'P', 'P', 'C', 'C', 'C',
             'C', 'C', 'Q', 'Q', 'RR', 'P', 'U', 'P', 'P', 'U', 'RD', 'U', 'C', 'P', 'P', 'P', 'P', 'P', 'C',
             'C', 'C', 'C', 'C', 'Q', 'Q', 'C', 'C', 'BR', 'P', 'U', 'P', 'P', 'U', 'BD', 'U', 'P', 'P', 'P',
             'P', 'P', 'C', 'C', 'C', 'C', 'Q', 'Q', 'input'],
            ['output'],
            name='a2_from_a1slice',
            domain='',
            equation='...a,br,cba,dr,ew,fde,...g,hfg,ic,iA,cA,cA,hA,hA,ij,ik,ii,jo,ko,jj,kk,...M,Nr,ONM,Pr,Qw,RPQ,...S,ORS,TO,Tn,On,On,zn,zn,zX,zz,TU,TV,TT,Uo,Vo,...l,mr,pml,qr,sw,tqs,...u,ptu,vp,vW,pW,pW,zW,zW,zz,zz,vx,vy,vv,xo,yo,xx,yy,...B,Cr,DCB,Er,Fw,GEF,...H,IGH,IZ,IZ,DZ,DZ,JZ,JI,JK,JL,JJ,Ko,Lo,...Yrw->...orw',
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
            _tensor('Q', TensorProto.FLOAT, (2, 10), INIT_Q_1),
            _tensor('U', TensorProto.FLOAT, (2, 2, 2), INIT_U_2),
            _tensor('C', TensorProto.FLOAT, (2, 2), INIT_C_3),
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
