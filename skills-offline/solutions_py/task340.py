from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task340'
TASK_NUM = 340
KAGGLE = {'score': 19.498742, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 245
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 9
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'precomp_factor'
OPSETS = [('', 14)]


# basis: FLOAT[4, 30], 120 value(s)
INIT_BASIS_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
 16.0, 17.0, 18.0, 19.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0]

# sideC: FLOAT[4, 4], 16 value(s)
INIT_SIDEC_1 = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0]

# sideR: FLOAT[4, 4], 16 value(s)
INIT_SIDER_2 = [0.0, 0.0, 0.0, 5.384944199533434e-16, 0.0, 0.0, 0.0, 5.384944199533434e-16, 0.0, 0.0, 5.397868463319355e-08, 0.0,
 -5.384944712361062e-10, 0.0, 5.384949708364672e-10, 0.0]

# maskCh: FLOAT[10], 10 value(s)
INIT_MASKCH_3 = [1.0, 28183830.0, 28183830.0, 28183830.0, 28183830.0, 28183830.0, 28183830.0, 28183830.0, 28183830.0, 28183830.0]

# g0: FLOAT[4, 3], 12 value(s)
INIT_G0_4 = [0.0, 0.01600000075995922, 0.0, 1.0, 1.0160000324249268, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]

# g1: FLOAT[2, 3], 6 value(s)
INIT_G1_5 = [1.0, 1.031999945640564, 0.0, 1.0, 1.0160000324249268, 1.0]

# g2: FLOAT[4, 3], 12 value(s)
INIT_G2_6 = [-62.5, 62.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 9.999999747378752e-05]

# srcpat_g2_core: FLOAT[4, 3, 2], 24 value(s)
INIT_SRCPAT_G2_CORE_7 = [0.0, 0.0, 0.01600000075995922, 0.01600000075995922, 1.0, 1.0, 0.0, 1.0, -0.01600000075995922, 0.984000027179718, 0.0,
 0.0, 0.0, 0.0, 0.01600000075995922, 0.01600000075995922, 0.0, 0.0, 0.0, 0.0, -159.98399353027344, 0.01600000075995922,
 0.0, 0.0]

# A: FLOAT[2, 4], 8 value(s)
INIT_A_8 = [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

# C: FLOAT[4, 3], 12 value(s)
INIT_C_9 = [0.0, 0.0, 1.0, 465.0, -99.9998779296875, 0.0, -43.04999923706055, 41.66660690307617, -100.0, -84.00988006591797,
 83.33321380615234, 0.0]

# Vcolor: FLOAT[3, 3], 9 value(s)
INIT_VCOLOR_10 = [-0.1562187522649765, -100.13629150390625, -123.9751968383789, 0.1562187522649765, 100.13621520996094,
 124.9749984741211, -4.999999969612645e-09, 0.007999994792044163, 0.0]


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
        # 0: Einsum inputs=54 outputs=1
        _node(
            'Einsum',
            ['basis', 'input', 'basis', 'sideR', 'A', 'A', 'basis', 'input', 'basis', 'sideR', 'basis', 'input',
             'basis', 'sideC', 'sideR', 'maskCh', 'basis', 'input', 'basis', 'sideR', 'sideC', 'maskCh', 'A',
             'srcpat_g2_core', 'g2', 'basis', 'input', 'basis', 'g1', 'g0', 'g2', 'srcpat_g2_core', 'g2', 'C',
             'g0', 'sideC', 'Vcolor', 'g0', 'C', 'g0', 'g2', 'g0', 'g1', 'g2', 'A', 'basis', 'input', 'basis',
             'sideR', 'basis', 'input', 'basis', 'sideC', 'maskCh'],
            ['output'],
            name='precomp_factor',
            domain='',
            equation='YT,...NST,US,UU,QY,PF,Fr,...erv,Gv,GG,jb,...dab,ia,sj,si,d,IA,...EAB,JB,tJ,tI,E,qK,Rnq,tn,Ky,...Eyw,lw,Qg,Wg,lg,Wup,su,zX,WX,zm,ZO,mZ,VH,RH,VO,Ro,Po,fo,pk,kx,...dhx,fh,zL,LC,...cCD,MD,zM,c->...chw',
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
            _tensor('basis', TensorProto.FLOAT, (4, 30), INIT_BASIS_0),
            _tensor('sideC', TensorProto.FLOAT, (4, 4), INIT_SIDEC_1),
            _tensor('sideR', TensorProto.FLOAT, (4, 4), INIT_SIDER_2),
            _tensor('maskCh', TensorProto.FLOAT, (10,), INIT_MASKCH_3),
            _tensor('g0', TensorProto.FLOAT, (4, 3), INIT_G0_4),
            _tensor('g1', TensorProto.FLOAT, (2, 3), INIT_G1_5),
            _tensor('g2', TensorProto.FLOAT, (4, 3), INIT_G2_6),
            _tensor('srcpat_g2_core', TensorProto.FLOAT, (4, 3, 2), INIT_SRCPAT_G2_CORE_7),
            _tensor('A', TensorProto.FLOAT, (2, 4), INIT_A_8),
            _tensor('C', TensorProto.FLOAT, (4, 3), INIT_C_9),
            _tensor('Vcolor', TensorProto.FLOAT, (3, 3), INIT_VCOLOR_10),
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
