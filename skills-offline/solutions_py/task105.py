from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task105'
TASK_NUM = 105
KAGGLE = {'score': 20.281501, 'date': '2026-07-13'}
MEMORY_BYTES = 8
PARAMS = 104
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task105_inline_gauge_fixed_A_diagonal_U'
OPSETS = [('', 18)]


# C: FLOAT[30, 2], 60 value(s)
INIT_C_0 = [1.0, 0.06666667014360428, 1.0, 0.13333334028720856, 1.0, 0.20000000298023224, 1.0, 0.2666666805744171, 1.0,
 0.3333333432674408, 1.0, 0.4000000059604645, 1.0, 0.46666666865348816, 1.0, 0.5333333611488342, 1.0,
 0.6000000238418579, 1.0, 0.6666666865348816, 1.0, 0.7333333492279053, 1.0, 0.800000011920929, 1.0, 0.8666666746139526,
 -1.7003751993179321, 0.7084896564483643, 1.7780095338821411, -0.8466712236404419, -1.7276073694229126,
 0.9597818851470947, 1.5767320394515991, -1.051154613494873, -1.342777967453003, 1.1189815998077393, 1.043468952178955,
 -1.159409999847412, -0.6993799805641174, 1.1656333208084106, 0.3367248773574829, -1.1224162578582764,
 0.3071720004081726, 1.0239067077636719, 0.6065096259117126, 1.0108493566513062, -0.9739490151405334,
 -1.0821655988693237, 1.2814528942108154, 1.0678774118423462, -1.5324262380599976, -1.0216175317764282,
 1.6937837600708008, 0.9409909844398499, -1.7531235218048096, -0.8348207473754883, 1.6822354793548584,
 0.7009314298629761, -0.9333333373069763, -0.46666666865348816]

# E: FLOAT[10, 2], 20 value(s)
INIT_E_1 = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Rel: FLOAT[2, 2, 2], 8 value(s)
INIT_REL_2 = [1.0, 0.0, 0.0, 0.0, 0.5, -15.0, 15.0, 0.0]

# A: FLOAT[2, 2, 2, 2], 16 value(s)
INIT_A_3 = [0.25, 31.211156845092773, 0.12903515994548798, -10.53176212310791, 0.005764749366790056, -2.0904955863952637,
 7.965923578012735e-05, 0.01582512818276882, 0.44406816363334656, 71.1953125, -0.12755799293518066, -5.79504919052124,
 -0.07544752955436707, 0.39932432770729065, 0.00662506278604269, 0.16749940812587738]


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
        # 0: Einsum inputs=55 outputs=1
        _node(
            'Einsum',
            ['E', 'A', 'Rel', 'A', 'Rel', 'E', 'input', 'C', 'C', 'C', 'Rel', 'A', 'C', 'Rel', 'A', 'C', 'Rel',
             'A', 'C', 'Rel', 'A', 'C', 'Rel', 'A', 'C', 'Rel', 'A', 'C', 'Rel', 'A', 'A', 'Rel', 'C', 'A',
             'Rel', 'C', 'A', 'Rel', 'C', 'C', 'Rel', 'A', 'C', 'Rel', 'A', 'A', 'Rel', 'C', 'C', 'C', 'C',
             'Rel', 'Rel', 'A', 'A'],
            ['T'],
            name='',
            domain='',
            equation='dp,eeee,pef,gggg,pgi,cp,nchw,hf,hi,hb,pab,aaaa,hk,pjk,jjjj,hq,poq,oooo,hm,plm,llll,hx,pvx,vvvv,hs,prs,rrrr,hA,pzA,zzzz,tttt,ptu,hu,BBBB,pBC,hC,HHHH,pHI,hI,hK,pJK,JJJJ,hE,pDE,DDDD,LLLL,pLM,hM,hy,hG,hO,pNO,pFG,FFFF,NNNN->y',
        ),
        # 1: Einsum inputs=145 outputs=1
        _node(
            'Einsum',
            ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'E', 'C', 'C', 'C', 'C',
             'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'input', 'E', 'E', 'E', 'E',
             'input', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
             'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'input', 'C', 'C',
             'input', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
             'Rel', 'Rel', 'Rel', 'Rel', 'A', 'A', 'Rel', 'C', 'C', 'C', 'C', 'input', 'E', 'E', 'input', 'C',
             'E', 'input', 'input', 'E', 'Rel', 'C', 'T', 'Rel', 'C', 'T', 'Rel', 'C', 'Rel', 'T', 'C', 'Rel',
             'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'Rel', 'E'],
            ['output'],
            name='',
            domain='',
            equation='Ye,Ye,Yf,Yf,Yf,Yf,Yk,Yk,Yk,Yk,Yk,Yk,Yk,Yk,Yd,PO,SO,SO,SO,SO,SO,SO,SO,SO,SO,SO,SO,SO,SO,SO,SO,SO,SO,Sz,...nST,nO,nO,nO,nO,...nWX,XO,XO,XO,XO,XO,XO,XO,XO,XO,XO,XO,XO,XO,XO,XO,XO,XO,XM,UO,UO,UO,UO,UO,UO,UO,UO,UO,UO,UO,UO,UO,UO,UO,UO,...nUV,UO,UC,...nQR,QO,QO,QO,QO,QO,QO,QO,QO,QO,QO,QO,QO,QO,QO,QO,QO,QO,Qv,fMN,bvx,byz,iCD,abik,mmmm,amq,hD,hx,hy,lq,...jhl,ja,Ed,...EFw,wN,Ge,...GHw,...chw,cp,fIJ,wI,B,iAB,hA,u,btu,ht,brs,r,hs,JKL,LLLL,LLLL,LLLL,LLLL,LLLL,LLLL,LLLL,KKKK,KKKK,KKKK,KKKK,KKKK,KKKK,KKKK,KKKK,pgp,op->...ohw',
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
            _tensor('C', TensorProto.FLOAT, (30, 2), INIT_C_0),
            _tensor('E', TensorProto.FLOAT, (10, 2), INIT_E_1),
            _tensor('Rel', TensorProto.FLOAT, (2, 2, 2), INIT_REL_2),
            _tensor('A', TensorProto.FLOAT, (2, 2, 2, 2), INIT_A_3),
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
