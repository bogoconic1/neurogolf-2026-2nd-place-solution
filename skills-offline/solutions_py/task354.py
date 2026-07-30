from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task354'
TASK_NUM = 354
KAGGLE = {'score': 20.751505, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 70
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 18)]


# Q: FLOAT[2, 30], 60 value(s)
INIT_Q_0 = [0.509056806564331, 0.5313015580177307, 0.6774421334266663, 0.8095648884773254, 0.765286386013031, 0.7533111572265625,
 0.5729413628578186, 0.3030253052711487, 0.09986885637044907, 0.10229231417179108, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5994032621383667, -0.595031201839447,
 -0.4678085446357727, -0.055175554007291794, 0.17558278143405914, 0.2928861081600189, 0.552823007106781,
 0.7532464861869812, 0.7849761843681335, 0.7826564908027649, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# R: FLOAT[10], 10 value(s)
INIT_R_1 = [1.0, -0.0010000000474974513, -0.0010000000474974513, -0.0010000000474974513, -0.0010000000474974513,
 -3.715988841390044e-29, -0.0010000000474974513, -0.0010000000474974513, -0.0010000000474974513,
 -0.0010000000474974513]


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
        # 0: Einsum inputs=90 outputs=1
        _node(
            'Einsum',
            ['Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q',
             'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q',
             'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q',
             'input', 'input', 'input', 'input', 'input', 'R', 'input', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q',
             'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'R', 'R', 'R'],
            ['output'],
            name='rank2poly',
            domain='',
            equation='ga,gd,ha,hd,ia,id,ja,jd,la,ld,ma,md,oa,od,pa,pd,qa,qd,sa,sd,td,te,vd,ve,wd,we,xd,xe,yd,ye,zd,ze,Ad,Ae,Bd,Be,Cd,Ce,Dd,De,Ee,Ef,Fe,Ff,Ge,Gf,He,Hf,Ie,If,Je,Jf,Ke,Kf,Le,Lf,Me,Mf,Ne,Nf,bnrd,bnre,bnra,bkua,bnrf,n,bnrc,Of,Oc,Pf,Pc,Qf,Qc,Rf,Rc,Sf,Sc,Tf,Tc,Uf,Uc,Vf,Vc,Wf,Wc,Xf,Xc,k,k,k->bkrc',
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
            _tensor('Q', TensorProto.FLOAT, (2, 30), INIT_Q_0),
            _tensor('R', TensorProto.FLOAT, (10,), INIT_R_1),
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
