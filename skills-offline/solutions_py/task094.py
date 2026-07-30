from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task094'
TASK_NUM = 94
KAGGLE = {'score': 20.375027, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 102
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'qfact'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'qfact'
OPSETS = [('', 14)]


# G: FLOAT[2, 30], 60 value(s)
INIT_G_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]

# C: FLOAT[10, 2], 20 value(s)
INIT_C_1 = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0]

# E: FLOAT[3, 2, 2], 12 value(s)
INIT_E_2 = [-37.38410949707031, -0.022891808301210403, 0.13407976925373077, 3.941479735658504e-05, -22.68157196044922,
 -4.41274078909204e-14, -2.4962725029414878e-14, -2.6246363631798886e-05, 38.881290435791016, -0.023808592930436134,
 0.13944947719573975, -4.0993301809066907e-05]

# QA: FLOAT[3, 2], 6 value(s)
INIT_QA_3 = [-1.1022975444793701, -0.14247533679008484, 9.871122360229492, 18.828670501708984, 0.9798069596290588,
 0.12664516270160675]

# QB: FLOAT[2, 2], 4 value(s)
INIT_QB_4 = [5.723321437835693, -11.4839506149292, 7.851586818695068, -14.815794944763184]


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
        # 0: Einsum inputs=57 outputs=1
        _node(
            'Einsum',
            ['input', 'C', 'G', 'G', 'E', 'input', 'G', 'G', 'E', 'QA', 'QB', 'E', 'G', 'G', 'input', 'G', 'G',
             'E', 'input', 'G', 'G', 'E', 'QA', 'QB', 'E', 'G', 'G', 'input', 'G', 'G', 'E', 'input', 'G', 'G',
             'E', 'QA', 'QB', 'E', 'G', 'G', 'input', 'G', 'G', 'E', 'input', 'G', 'G', 'E', 'QA', 'QB', 'E',
             'G', 'G', 'input', 'C', 'C', 'C'],
            ['output'],
            name='qfact',
            domain='',
            equation='...art,as,vr,xr,bvx,...ayz,Ay,By,bAB,bg,ug,bgi,gh,ih,...aCD,EC,FC,dEF,...aGH,IG,JG,dIJ,dj,uj,djl,jh,lh,...aLK,MK,NK,eMN,...aPO,QO,RO,eQR,em,um,emn,mw,nw,...aTS,US,VS,fUV,...aXW,YW,ZW,fYZ,fo,uo,foq,ow,qw,...chw,cp,kp,cu->...khw',
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
            _tensor('G', TensorProto.FLOAT, (2, 30), INIT_G_0),
            _tensor('C', TensorProto.FLOAT, (10, 2), INIT_C_1),
            _tensor('E', TensorProto.FLOAT, (3, 2, 2), INIT_E_2),
            _tensor('QA', TensorProto.FLOAT, (3, 2), INIT_QA_3),
            _tensor('QB', TensorProto.FLOAT, (2, 2), INIT_QB_4),
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
