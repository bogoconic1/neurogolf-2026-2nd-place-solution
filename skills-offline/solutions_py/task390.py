from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task390'
TASK_NUM = 390
KAGGLE = {'score': 20.212508, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 120
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 22)]


# P: FLOAT[30, 3], 90 value(s)
INIT_P_0 = [0.16297979652881622, -2.4794907569885254, 0.16514688730239868, 0.10062427073717117, 0.8878211975097656,
 0.47296300530433655, 1.3219949007034302, 1.287104606628418, -0.01085700374096632, -0.27360478043556213,
 -0.05928066000342369, 2.1562063694000244, -1.3421015739440918, -0.1163247674703598, 1.3177045583724976,
 -0.0300806425511837, 0.09836924076080322, 0.9247319102287292, -0.15987730026245117, 0.2992963194847107,
 -0.10650691390037537, 0.3863479793071747, 0.682950496673584, 0.5917626619338989, -1.3516697883605957,
 0.09465375542640686, 0.4928532838821411, 0.6972298622131348, 0.7490824460983276, -0.04036674648523331,
 -0.7700775265693665, 0.3962060511112213, 1.8091378211975098, 0.033956799656152725, 0.10762812197208405,
 1.075068712234497, 1.1724135875701904, -0.3845135569572449, -0.09386840462684631, 0.017140964046120644,
 0.48065024614334106, -0.018286854028701782, -2.6432764530181885, 0.5521516799926758, -0.7203196883201599, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# C: FLOAT[2, 10], 20 value(s)
INIT_C_1 = [1.0, -1.384615421295166, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9230769276618958, 1.0, 0.0, 0.0, -1.0, 0.0,
 0.0, 0.0, 0.0]

# U: FLOAT[2, 2], 4 value(s)
INIT_U_2 = [-1.4881738424301147, -1.5697283744812012, -1.2478134632110596, 1.2039891481399536]

# S: FLOAT[2, 3], 6 value(s)
INIT_S_3 = [-0.3311775028705597, 0.040360331535339355, 0.7805949449539185, -1.1609001159667969, 0.10191048681735992,
 2.7210910320281982]


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
        # 0: Einsum inputs=96 outputs=1
        _node(
            'Einsum',
            ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
             'S', 'U', 'U', 'C', 'P', 'input', 'C', 'input', 'P', 'S', 'P', 'input', 'C', 'U', 'U', 'C',
             'input', 'P', 'S', 'S', 'S', 'S', 'P', 'P', 'U', 'U', 'C', 'input', 'P', 'C', 'input', 'P', 'P',
             'S', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
             'C', 'S', 'P', 'P', 'input', 'C', 'P', 'input', 'C', 'U', 'U', 'input', 'C', 'C', 'C', 'C', 'C',
             'C', 'C', 'C', 'C', 'C', 'C'],
            ['output'],
            name='',
            domain='',
            equation='tI,tI,tI,tI,tI,tI,tI,tI,tI,tI,tI,tI,tI,tI,tI,tI,tI,hI,hI,hI,hk,ze,zf,fl,pk,nlps,ej,njqs,qk,tx,wx,nuws,au,oa,ob,bv,nvys,yx,oR,oR,OR,OR,rk,rx,OB,OA,AU,nUrW,WX,BV,nVrY,YX,sX,MX,MN,MN,MN,MN,MN,MN,MN,MN,MN,MN,MN,MN,MN,MN,MN,MN,MN,HN,HN,HN,HK,sK,QK,nJrQ,EJ,PK,nLrP,FL,ZF,ZE,nirs,Ti,Ti,Si,Sc,Tc,dc,dC,mc,mG,gD,gc->ncrs',
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
            _tensor('P', TensorProto.FLOAT, (30, 3), INIT_P_0),
            _tensor('C', TensorProto.FLOAT, (2, 10), INIT_C_1),
            _tensor('U', TensorProto.FLOAT, (2, 2), INIT_U_2),
            _tensor('S', TensorProto.FLOAT, (2, 3), INIT_S_3),
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
