from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task093'
TASK_NUM = 93
KAGGLE = {'score': 20.617973, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task093_p80_inline'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task093_p80_inline'
OPSETS = [('', 20)]


# ch: FLOAT[2, 10], 20 value(s)
INIT_CH_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -2.0, -1.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0]

# coord_code: FLOAT[2, 30], 60 value(s)
INIT_COORD_CODE_1 = [0.26725122332572937, 0.1719139814376831, -0.07305468618869781, 0.028958143666386604, -0.133949413895607,
 -0.24184420704841614, 0.3525705635547638, -1.614066243171692, 1.611830472946167, -1.6088954210281372, 1.60586416721344,
 -1.6031606197357178, 1.6013303995132446, 1.6010860204696655, -1.156943917274475, -0.2663784623146057,
 -2.590996265411377, -0.2664458453655243, 1.6049052476882935, -0.26615217328071594, 1.4918705224990845,
 -2.408069610595703, -2.7722835540771484, -1.8632500171661377, 2.8211042881011963, -2.123652935028076,
 2.4343674182891846, 2.331331968307495, -0.26643696427345276, 2.331195831298828, -1.7873315811157227,
 -1.7626326084136963, 1.7387622594833374, 1.715022087097168, -1.6908226013183594, -1.665569543838501,
 1.6393970251083374, 0.04374480992555618, -0.1631680130958557, 0.27713167667388916, -0.3889596462249756,
 0.4976602792739868, -0.6032131314277649, -0.7054301500320435, 2.179727554321289, 1.2439613342285156, 1.58176851272583,
 1.2438020706176758, -2.275623083114624, 1.2440110445022583, 0.8050631880760193, 1.5123169422149658, 0.6555759310722351,
 -0.9006364345550537, -1.0807987451553345, 2.1922829151153564, -0.03135250136256218, -1.879717230796814,
 1.2439870834350586, -1.8789596557617188]


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
        # 0: Einsum inputs=67 outputs=1
        _node(
            'Einsum',
            ['coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code',
             'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code',
             'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code',
             'coord_code', 'coord_code', 'ch', 'ch', 'ch', 'ch', 'input', 'input', 'coord_code', 'coord_code',
             'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code',
             'coord_code', 'coord_code', 'ch', 'ch', 'input', 'coord_code', 'coord_code', 'coord_code',
             'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code',
             'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code', 'ch', 'input', 'ch',
             'coord_code', 'coord_code', 'coord_code', 'coord_code', 'coord_code', 'ch'],
            ['output'],
            name='',
            domain='',
            equation='ia,ia,ia,ia,Aa,iV,iV,iV,jV,jV,jV,jV,jV,jV,EV,FV,jh,jc,jc,jc,jc,Bc,Bb,Be,je,Ad,id,ndrz,nerq,Eq,Fq,lm,lt,lt,lt,lt,Ht,Hx,Es,Fs,Hg,lg,ngps,Kp,Lp,Kr,Lr,KW,LW,lW,lW,lW,lW,lW,lW,kW,kW,kW,kf,nfys,Gf,Gu,ku,ku,ku,ku,Mo->nors',
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
            _tensor('ch', TensorProto.FLOAT, (2, 10), INIT_CH_0),
            _tensor('coord_code', TensorProto.FLOAT, (2, 30), INIT_COORD_CODE_1),
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
