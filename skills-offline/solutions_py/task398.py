from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task398'
TASK_NUM = 398
KAGGLE = {'score': 19.652892, 'date': '2026-07-13'}
MEMORY_BYTES = 24
PARAMS = 186
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task398_updated_v9_no_S3_integer_scale_direct_signed_logits'
OPSETS = [('', 18)]


# U: FLOAT[2, 2, 30], 120 value(s)
INIT_U_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.001953125, -0.001953125, -0.001953125, -0.001953125,
 -0.001953125, -0.00390625, -0.00390625, -0.00390625, -0.00390625, -0.00390625, -0.005859375, -0.005859375,
 -0.005859375, -0.005859375, -0.005859375, -0.0078125, -0.0078125, -0.0078125, -0.0078125, -0.0078125, 0.0, 0.0, 0.0,
 0.0, 0.0, 1.0, -0.015640322118997574, -0.9995107650756836, 0.04690566658973694, 0.9980435371398926,
 -0.07812511175870895, -0.9955997467041016, 0.10926811397075653, 0.9921817183494568, -0.14030419290065765,
 -0.9877929091453552, 0.17120300233364105, 0.9824376106262207, -0.20193427801132202, -0.9761209487915039,
 0.23246797919273376, 0.9688491821289062, -0.26277419924736023, -0.9606294631958008, 0.29282331466674805,
 0.9514697790145874, -0.32258591055870056, -0.9413790702819824, 0.3520328402519226, 0.9303672313690186, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.9998776912689209, -0.03127681836485863, -0.9988993406295776, 0.06252303719520569, 0.9969435334205627,
 -0.09370807558298111, -0.9940122961997986, 0.12480141967535019, 0.9901084303855896, -0.15577265620231628,
 -0.9852357506752014, 0.18659146130084991, 0.9793990850448608, -0.21722769737243652, -0.9726040363311768,
 0.2476513832807541, 0.9648573398590088, -0.2778327465057373, -0.9561665654182434, 0.30774223804473877,
 0.9465401768684387, -0.33735063672065735, -0.935987651348114, 0.3666289448738098, 0.0, 0.0, 0.0, 0.0, 0.0]

# core: FLOAT[2, 2, 2], 8 value(s)
INIT_CORE_1 = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0]

# C0: FLOAT[2, 2], 4 value(s)
INIT_C0_2 = [1.0, 512.0, -512.0, 0.0]

# S1: FLOAT[2], 2 value(s)
INIT_S1_3 = [2.0, 1.0]

# S2: FLOAT[2], 2 value(s)
INIT_S2_4 = [3.0, 1.0]

# P: FLOAT[3, 10], 30 value(s)
INIT_P_5 = [1.0, 2.235844373703003, 2.235844373703003, -2.235844373703003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001954843057319522, 0.0,
 0.0, 0.0, 2.235844373703003, 2.235844373703003, -2.235844373703003, 0.0, 0.0, 0.0, -1.0019547939300537, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 2.235844373703003, 2.235844373703003, -2.235844373703003]

# R: FLOAT[2, 10], 20 value(s)
INIT_R_6 = [-1.0, 0.0625, 0.0625, 0.0625, 0.5, 0.5, 0.5, -0.0625, -0.0625, -0.0625, 0.0, 0.0625, -0.0625, 0.125, 0.5, -0.5, 1.0,
 -0.0625, 0.0625, -0.125]


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
        # 0: Einsum inputs=3 outputs=1
        _node(
            'Einsum',
            ['input', 'P', 'P'],
            ['m_count'],
            name='',
            domain='',
            equation='nqxs,Aq,Bq->n',
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['m_count'],
            ['m_idx'],
            name='',
            domain='',
            to=6,
        ),
        # 2: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['U', 'm_idx'],
            ['phase'],
            name='',
            domain='',
            axis=2,
        ),
        # 3: Einsum inputs=80 outputs=1
        _node(
            'Einsum',
            ['S1', 'S1', 'S1', 'S1', 'S1', 'C0', 'S1', 'phase', 'phase', 'C0', 'S1', 'S1', 'core', 'S1',
             'phase', 'C0', 'S1', 'S2', 'phase', 'C0', 'S2', 'S1', 'phase', 'C0', 'S1', 'S2', 'phase', 'C0',
             'S2', 'phase', 'C0', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'C0', 'phase', 'core', 'S1', 'S1',
             'C0', 'U', 'core', 'U', 'phase', 'core', 'U', 'U', 'core', 'phase', 'core', 'U', 'U', 'U', 'U',
             'input', 'core', 'phase', 'U', 'core', 'U', 'phase', 'core', 'U', 'core', 'U', 'core', 'R', 'P',
             'R', 'P', 'P', 'R', 'R', 'R', 'R'],
            ['output'],
            name='',
            domain='',
            equation='X,X,X,X,P,XP,P,FX...,FX...,XY,Y,Y,FFF,T,FT...,TU,U,M,FM...,MN,N,K,FK...,KL,L,V,FV...,VW,W,FI...,IJ,FLr,FPr,FNr,FJr,FWc,FUc,FYc,FSc,RS,FR...,ZZF,Z,Z,za,agc,fge,afr,ab...,dbe,hmc,hlr,lmk,hi...,jik,ads,hjs,pus,pBs,...qxs,utv,pt...,pwr,wyv,pyc,pA...,BAC,pEc,DEC,pDr,OQp,Zq,Hq,Gq,nq,no,Go,Zq,Zo,Zo->...orc',
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
            _tensor('U', TensorProto.FLOAT, (2, 2, 30), INIT_U_0),
            _tensor('core', TensorProto.FLOAT, (2, 2, 2), INIT_CORE_1),
            _tensor('C0', TensorProto.FLOAT, (2, 2), INIT_C0_2),
            _tensor('S1', TensorProto.FLOAT, (2,), INIT_S1_3),
            _tensor('S2', TensorProto.FLOAT, (2,), INIT_S2_4),
            _tensor('P', TensorProto.FLOAT, (3, 10), INIT_P_5),
            _tensor('R', TensorProto.FLOAT, (2, 10), INIT_R_6),
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
