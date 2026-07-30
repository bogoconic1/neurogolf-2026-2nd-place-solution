from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task005'
TASK_NUM = 5
KAGGLE = {'score': 18.546375, 'date': '2026-07-15'}
MEMORY_BYTES = 44
PARAMS = 591
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task005_exact_rq_rank4'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task005_exact_eb_cj_rank2_rq_rank4_m44_p614'
OPSETS = [('', 22)]


# Q: FLOAT[30, 4], 120 value(s)
INIT_Q_0 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0]

# V02: FLOAT[3], 3 value(s)
INIT_V02_1 = [1.0, 0.0, 1.0]

# T: FLOAT[3, 3, 2, 3], 54 value(s)
INIT_T_2 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]

# M: FLOAT[3, 3, 2, 3], 54 value(s)
INIT_M_3 = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]

# C: FLOAT[10, 2], 20 value(s)
INIT_C_4 = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# POA: FLOAT[2, 2], 4 value(s)
INIT_POA_5 = [1.0, -1.0, -2000000.0, 2000001.0]

# BC: FLOAT[3, 7, 3], 63 value(s)
INIT_BC_6 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]

# P: FLOAT[10, 3], 30 value(s)
INIT_P_7 = [0.9999967813491821, 1.4142112731933594, 0.9999969601631165, -9.310931636719033e-05, 1.4141789674758911,
 -1.0001267194747925, 2.999959945678711, 2.8284003734588623, -4.000077724456787, 7.999959468841553, 4.242629051208496,
 -9.000049591064453, 14.99996566772461, 5.656846046447754, -16.000038146972656, 23.99997329711914, 7.071060657501221,
 -25.000030517578125, 34.999977111816406, 8.485274314880371, -36.000030517578125, 47.99997329711914, 9.899487495422363,
 -49.00003433227539, 62.999969482421875, 11.313700675964355, -64.00003814697266, 79.99996185302734, 12.727912902832031,
 -81.00004577636719]

# S1: FLOAT[4, 2], 8 value(s)
INIT_S1_8 = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0]

# S2: FLOAT[3, 2], 6 value(s)
INIT_S2_9 = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0]

# E3: FLOAT[30, 3], 90 value(s)
INIT_E3_10 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# E2: FLOAT[30, 2], 60 value(s)
INIT_E2_11 = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# GD: FLOAT[3], 3 value(s)
INIT_GD_12 = [0.0, 1.0, 1.0]

# RQA: FLOAT[7, 4], 28 value(s)
INIT_RQA_13 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 0.0, 0.0, 1.0, 0.0]

# RQB: FLOAT[4, 3, 4], 48 value(s)
INIT_RQB_14 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
 1.0]


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
        # 0: Einsum inputs=49 outputs=1
        _node(
            'Einsum',
            ['S2', 'V02', 'RQA', 'RQB', 'Q', 'input', 'S2', 'T', 'BC', 'S2', 'RQA', 'RQB', 'E3', 'E2', 'Q',
             'input', 'E3', 'E2', 'T', 'BC', 'GD', 'GD', 'V02', 'GD', 'BC', 'T', 'RQA', 'RQB', 'E3', 'E2', 'Q',
             'input', 'E3', 'E2', 'BC', 'T', 'GD', 'POA', 'C', 'input', 'Q', 'RQA', 'RQB', 'GD', 'RQA', 'RQB',
             'S1', 'GD', 'V02'],
            ['HY1'],
            name='HY1',
            domain='',
            equation='iB,i,yn,niL,lL,...clm,sG,RSds,CyR,CK,yq,qCT,HS,Hd,HT,...cHI,IV,Ie,UVet,DxU,t,D,D,u,DyX,XYfu,yr,rDZ,EY,Ef,EZ,...cEF,Fb,Fj,Jxa,abjv,v,Ap,cA,...cMN,Mg,yw,wJg,J,yz,zhk,ko,h,h->...o',
        ),
        # 1: Einsum inputs=48 outputs=1
        _node(
            'Einsum',
            ['S2', 'V02', 'RQA', 'RQB', 'Q', 'input', 'S2', 'T', 'BC', 'S2', 'RQA', 'RQB', 'E3', 'E2', 'Q',
             'input', 'E3', 'E2', 'T', 'BC', 'GD', 'GD', 'V02', 'GD', 'BC', 'T', 'RQA', 'RQB', 'E3', 'E2', 'Q',
             'input', 'E3', 'E2', 'BC', 'T', 'GD', 'POA', 'C', 'input', 'Q', 'RQA', 'RQB', 'GD', 'BC', 'S2',
             'GD', 'V02'],
            ['HY2'],
            name='HY2',
            domain='',
            equation='iB,i,yn,niL,lL,...clm,sG,RSds,CyR,CK,yq,qCT,HS,Hd,HT,...cHI,IV,Ie,UVet,DxU,t,D,D,u,DyX,XYfu,yr,rDZ,EY,Ef,EZ,...cEF,Fb,Fj,Jxa,abjv,v,Ap,cA,...cMN,Mg,yw,wJg,J,hyk,ko,h,h->...o',
        ),
        # 2: Einsum inputs=45 outputs=1
        _node(
            'Einsum',
            ['S2', 'V02', 'RQA', 'RQB', 'Q', 'input', 'Q', 'E3', 'E2', 'T', 'GD', 'S2', 'RQA', 'RQB', 'BC',
             'S2', 'input', 'Q', 'E3', 'E2', 'RQA', 'RQB', 'T', 'BC', 'GD', 'GD', 'V02', 'RQA', 'RQB', 'Q',
             'input', 'E3', 'E2', 'T', 'BC', 'S2', 'GD', 'POA', 'C', 'input', 'E3', 'E2', 'BC', 'T', 'S2'],
            ['HX'],
            name='HX',
            domain='',
            equation='iB,i,yn,niL,lL,...clm,mQ,mP,md,OPdr,r,rC,xo,ojQ,jxO,jG,...cHI,IW,IV,Ig,xq,qDW,UVgt,DxU,t,D,D,ys,sDZ,EZ,...cEF,Fb,Fh,abhv,Jxa,vK,v,Ap,cA,...cMN,Mf,Mk,Jye,efkw,wR->...x',
        ),
        # 3: Einsum inputs=68 outputs=1
        _node(
            'Einsum',
            ['GD', 'V02', 'HY1', 'C', 'S1', 'RQA', 'BC', 'S2', 'C', 'HY2', 'C', 'S1', 'S1', 'RQB', 'S2', 'GD',
             'RQB', 'S1', 'S1', 'S1', 'RQA', 'HX', 'S1', 'BC', 'C', 'S2', 'BC', 'S2', 'C', 'RQA', 'BC', 'RQB',
             'Q', 'E3', 'T', 'E2', 'input', 'E3', 'E2', 'Q', 'RQB', 'BC', 'T', 'C', 'P', 'S2', 'BC', 'P', 'S2',
             'P', 'POA', 'RQB', 'C', 'Q', 'input', 'Q', 'RQB', 'BC', 'T', 'M', 'M', 'T', 'E3', 'E2', 'Q', 'E3',
             'E2', 'Q'],
            ['output'],
            name='output',
            domain='',
            equation='j,j,...H,sH,lH,yl,jyE,EK,sK,...K,sq,mq,Iq,miI,iq,i,WiP,Pq,Wq,WM,JW,...J,mM,jJT,sM,TN,jxF,FN,sN,xm,ayB,laG,RG,Rn,BnDf,RD,...sRS,Se,SC,Sg,mbg,bxt,teCU,sA,sd,dA,uyO,od,jA,oj,AA,luQ,LA,hQ,...Lhw,wX,mvX,vxY,OVcf,OVck,YZpk,YZpU,zZ,zp,zX,rV,rc,rQ->...orz',
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
            _tensor('Q', TensorProto.FLOAT, (30, 4), INIT_Q_0),
            _tensor('V02', TensorProto.FLOAT, (3,), INIT_V02_1),
            _tensor('T', TensorProto.FLOAT, (3, 3, 2, 3), INIT_T_2),
            _tensor('M', TensorProto.FLOAT, (3, 3, 2, 3), INIT_M_3),
            _tensor('C', TensorProto.FLOAT, (10, 2), INIT_C_4),
            _tensor('POA', TensorProto.FLOAT, (2, 2), INIT_POA_5),
            _tensor('BC', TensorProto.FLOAT, (3, 7, 3), INIT_BC_6),
            _tensor('P', TensorProto.FLOAT, (10, 3), INIT_P_7),
            _tensor('S1', TensorProto.FLOAT, (4, 2), INIT_S1_8),
            _tensor('S2', TensorProto.FLOAT, (3, 2), INIT_S2_9),
            _tensor('E3', TensorProto.FLOAT, (30, 3), INIT_E3_10),
            _tensor('E2', TensorProto.FLOAT, (30, 2), INIT_E2_11),
            _tensor('GD', TensorProto.FLOAT, (3,), INIT_GD_12),
            _tensor('RQA', TensorProto.FLOAT, (7, 4), INIT_RQA_13),
            _tensor('RQB', TensorProto.FLOAT, (4, 3, 4), INIT_RQB_14),
        ],
        value_info=[
            _vi('HY1', TensorProto.FLOAT, [1, 2]),
            _vi('HY2', TensorProto.FLOAT, [1, 2]),
            _vi('HX', TensorProto.FLOAT, [1, 7]),
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
