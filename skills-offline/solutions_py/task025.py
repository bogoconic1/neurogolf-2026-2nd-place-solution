from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task025'
TASK_NUM = 25
KAGGLE = {'score': 18.571895, 'date': '2026-07-14'}
MEMORY_BYTES = 0
PARAMS = 619
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task025_expand_simplify_recompress_alias_GR_RVt'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task025_u_gu_factor_613_fold_g_v6618_global25'
OPSETS = [('', 18)]


# BP: FLOAT[3, 30], 90 value(s)
INIT_BP_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
 -1.0, 1.0, -1.0]

# RV0_L: FLOAT[3, 3], 9 value(s)
INIT_RV0_L_1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]

# R5: FLOAT[5, 30], 150 value(s)
INIT_R5_2 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

# K5: FLOAT[3, 5, 5], 75 value(s)
INIT_K5_3 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
 0.0, 0.0, 0.0]

# R3: FLOAT[3, 30], 90 value(s)
INIT_R3_4 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]

# K3: FLOAT[3, 3, 3], 27 value(s)
INIT_K3_5 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -0.0, -0.0, -0.10100000351667404, -0.10100000351667404, -0.0, -0.0, -0.0,
 -0.10100000351667404, -0.0, -0.0, -0.10100000351667404, -0.0, -0.0, -0.0, -0.10100000351667404, -0.10100000351667404,
 -0.0, -0.0]

# PMA: FLOAT[3, 3], 9 value(s)
INIT_PMA_6 = [1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0, -2.0, -2.0]

# KP2: FLOAT[3, 3], 9 value(s)
INIT_KP2_7 = [0.5, 0.0, 0.5, 0.5, 0.0, -0.5, 0.5, 0.0, -0.5]

# R4: FLOAT[10, 3], 30 value(s)
INIT_R4_8 = [3.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0,
 -2.0, 1.0, 1.0, -2.0, 1.0, 1.0, -2.0]

# S4: FLOAT[10, 4], 40 value(s)
INIT_S4_9 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

# H: FLOAT[4, 4, 3], 48 value(s)
INIT_H_10 = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1000000.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
 0.0, 1.0, 1.0, -1000000.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1000000.0, 1.0, 0.0, 0.0, 1.0, 0.0,
 0.0, 1.0, 0.0, 1.0, 1.0]

# M: FLOAT[3, 3], 9 value(s)
INIT_M_11 = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]

# E3: FLOAT[3, 3], 9 value(s)
INIT_E3_12 = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.25, 0.0]

# E5: FLOAT[3, 5], 15 value(s)
INIT_E5_13 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

# fold_B: FLOAT[3, 3], 9 value(s)
INIT_FOLD_B_14 = [1.0, 0.0, 0.0, 1.0, 4.0, -4.0, 1.0, 0.0, 0.0]


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
        # 0: Einsum inputs=81 outputs=1
        _node(
            'Einsum',
            ['E3', 'PMA', 'PMA', 'PMA', 'E3', 'KP2', 'PMA', 'PMA', 'RV0_L', 'KP2', 'E3', 'PMA', 'E5', 'M',
             'PMA', 'E3', 'R3', 'RV0_L', 'R5', 'BP', 'E5', 'input', 'R5', 'input', 'R3', 'PMA', 'BP', 'R5',
             'K3', 'KP2', 'PMA', 'PMA', 'K5', 'PMA', 'R5', 'KP2', 'R3', 'M', 'RV0_L', 'E3', 'BP', 'KP2', 'E3',
             'E5', 'R5', 'R3', 'E5', 'R5', 'input', 'input', 'input', 'BP', 'E3', 'PMA', 'PMA', 'fold_B', 'R4',
             'S4', 'H', 'S4', 'R4', 'M', 'S4', 'H', 'S4', 'RV0_L', 'BP', 'PMA', 'BP', 'R3', 'R5', 'K5', 'K3',
             'KP2', 'R5', 'R3', 'BP', 'BP', 'input', 'R4', 'R4'],
            ['output'],
            name='output',
            domain='',
            equation='tO,tT,tO,PO,PO,PP,Tj,tT,jf,PP,tT,tO,PA,Ot,PO,PV,Vy,OO,Ay,fm,OY,nbmy,Yx,nbmx,Mm,jj,Um,Gm,WLM,WU,Wj,je,WEG,je,Eh,NN,Lh,tN,tF,je,Uh,NN,Fk,Fp,pu,ku,Nq,qv,navl,naul,nahc,zc,iz,iz,iz,ti,aD,aX,Xgt,og,oD,Bt,bR,RdB,od,is,sl,Zi,Ql,Il,Kl,ZJK,ZHI,ZQ,Jw,Hw,Qw,er,nbrw,bC,oC->nohw',
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
            _tensor('BP', TensorProto.FLOAT, (3, 30), INIT_BP_0),
            _tensor('RV0_L', TensorProto.FLOAT, (3, 3), INIT_RV0_L_1),
            _tensor('R5', TensorProto.FLOAT, (5, 30), INIT_R5_2),
            _tensor('K5', TensorProto.FLOAT, (3, 5, 5), INIT_K5_3),
            _tensor('R3', TensorProto.FLOAT, (3, 30), INIT_R3_4),
            _tensor('K3', TensorProto.FLOAT, (3, 3, 3), INIT_K3_5),
            _tensor('PMA', TensorProto.FLOAT, (3, 3), INIT_PMA_6),
            _tensor('KP2', TensorProto.FLOAT, (3, 3), INIT_KP2_7),
            _tensor('R4', TensorProto.FLOAT, (10, 3), INIT_R4_8),
            _tensor('S4', TensorProto.FLOAT, (10, 4), INIT_S4_9),
            _tensor('H', TensorProto.FLOAT, (4, 4, 3), INIT_H_10),
            _tensor('M', TensorProto.FLOAT, (3, 3), INIT_M_11),
            _tensor('E3', TensorProto.FLOAT, (3, 3), INIT_E3_12),
            _tensor('E5', TensorProto.FLOAT, (3, 5), INIT_E5_13),
            _tensor('fold_B', TensorProto.FLOAT, (3, 3), INIT_FOLD_B_14),
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
