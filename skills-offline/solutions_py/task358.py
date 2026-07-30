from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task358'
TASK_NUM = 358
KAGGLE = {'score': 19.68188, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 204
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'expand_binary_state_implicit_permutation'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'binary_state_no_tp'
OPSETS = [('', 16)]


# CB: FLOAT[4, 30], 120 value(s)
INIT_CB_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 -0.7118878960609436, -0.5511264204978943, -0.0725911483168602, -2.0323712825775146, -1.7324930429458618,
 -0.38569188117980957, -0.012141341343522072, -0.2908346652984619, -1.7961395978927612, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 1.0, 0.8660253882408142, 0.5, 0.0, -0.5, -0.8660253882408142, -1.0, -0.8660253882408142, -0.5, 0.0, 0.5,
 0.8660253882408142, 1.0, 0.8660253882408142, 0.5, 0.0, -0.5, -0.8660253882408142, -1.0, -0.8660253882408142, -0.5,
 -0.47720444202423096, -0.9563425183296204, 0.15099163353443146, -1.5484379529953003, 0.5063915252685547,
 -0.3139646649360657, 0.7738189697265625, -0.4226768910884857, 1.815128207206726, 0.0, 0.5, 0.8660253882408142, 1.0,
 0.8660253882408142, 0.5, 0.0, -0.5, -0.8660253882408142, -1.0, -0.8660253882408142, -0.5, 0.0, 0.5, 0.8660253882408142,
 1.0, 0.8660253882408142, 0.5, 0.0, -0.5, -0.8660253882408142, 0.3332248628139496, 1.1824703216552734,
 0.1877318173646927, 1.1712031364440918, -1.6310441493988037, 0.2418990284204483, 1.2435945272445679,
 0.4775562286376953, -1.2131224870681763]

# LR2: FLOAT[2, 2, 2, 2], 16 value(s)
INIT_LR2_1 = [0.5, -0.5, 0.5, -0.5, 3.0, -1.0, -2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# E: FLOAT[4, 10], 40 value(s)
INIT_E_2 = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
 0.7660444378852844, 0.1736481785774231, -0.5, -0.9396926164627075, -0.9396926164627075, -0.5, 0.1736481785774231,
 0.7660444378852844, 0.0, 0.0, 0.6427876353263855, 0.9848077297210693, 0.8660253882408142, 0.3420201539993286,
 -0.3420201539993286, -0.8660253882408142, -0.9848077297210693, -0.6427876353263855]

# Q0: FLOAT[4], 4 value(s)
INIT_Q0_3 = [-0.5199999809265137, 0.0, 1.0, 1.0]

# Q3: FLOAT[2, 4], 8 value(s)
INIT_Q3_4 = [1.866025447845459, 0.0, 3.732050895690918, 3.732050895690918, -2.732050895690918, -0.0, -2.732050895690918,
 -2.732050895690918]

# HSEL: FLOAT[2, 4], 8 value(s)
INIT_HSEL_5 = [0.0, 0.5, 0.0, 0.0, 1.0, -0.5, 0.0, 0.0]

# HCG: FLOAT[2, 4], 8 value(s)
INIT_HCG_6 = [0.0, 1.0, 0.0, 0.0, -4.55555534362793, 0.0, 5.55555534362793, 5.55555534362793]


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
        # 0: Einsum inputs=80 outputs=1
        _node(
            'Einsum',
            ['Q0', 'Q0', 'Q0', 'Q0', 'Q0', 'Q3', 'CB', 'CB', 'CB', 'CB', 'Q0', 'CB', 'CB', 'CB', 'CB', 'Q0',
             'CB', 'CB', 'Q3', 'Q3', 'CB', 'CB', 'input', 'E', 'CB', 'CB', 'CB', 'Q3', 'CB', 'CB', 'Q3', 'CB',
             'CB', 'CB', 'CB', 'Q3', 'Q0', 'CB', 'CB', 'CB', 'CB', 'Q0', 'CB', 'input', 'E', 'Q0', 'Q0', 'CB',
             'Q0', 'Q0', 'Q0', 'E', 'input', 'CB', 'HSEL', 'HCG', 'LR2', 'HSEL', 'E', 'input', 'E', 'E', 'HCG',
             'E', 'HSEL', 'LR2', 'LR2', 'LR2', 'HSEL', 'CB', 'input', 'E', 'HSEL', 'CB', 'input', 'E', 'HSEL',
             'CB', 'input', 'E'],
            ['output'],
            name='output',
            domain='',
            equation='F,F,F,F,F,QU,Fr,Ur,Ug,Fg,s,sg,sr,Og,Or,O,Rr,Rg,pR,TZ,Zr,Zg,...dgc,yd,Zn,Zn,En,SE,Ec,Dc,PD,Db,Eb,Cc,Cb,pC,e,ec,eb,Nb,Nc,N,fb,...arb,xa,f,f,fc,f,f,f,Xl,...lrm,Xm,MX,Mx,tMpz,zV,Vu,...urc,xk,yk,ty,Wk,wW,tMpw,tMpo,MtpJ,oY,Yv,...qrv,Yq,th,hI,...HIc,hH,Ji,iL,...KLc,iK->...krc',
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
            _tensor('CB', TensorProto.FLOAT, (4, 30), INIT_CB_0),
            _tensor('LR2', TensorProto.FLOAT, (2, 2, 2, 2), INIT_LR2_1),
            _tensor('E', TensorProto.FLOAT, (4, 10), INIT_E_2),
            _tensor('Q0', TensorProto.FLOAT, (4,), INIT_Q0_3),
            _tensor('Q3', TensorProto.FLOAT, (2, 4), INIT_Q3_4),
            _tensor('HSEL', TensorProto.FLOAT, (2, 4), INIT_HSEL_5),
            _tensor('HCG', TensorProto.FLOAT, (2, 4), INIT_HCG_6),
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
