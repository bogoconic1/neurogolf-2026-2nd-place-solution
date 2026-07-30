from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task206'
TASK_NUM = 206
KAGGLE = {'score': 19.900134, 'date': '2026-07-15'}
MEMORY_BYTES = 32
PARAMS = 132
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task206_expand_simplify_recompress_m212'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task206_updated_v9'
OPSETS = [('', 16)]


# px0: FLOAT[2, 2, 2], 8 value(s)
INIT_PX0_0 = [-18.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# px2: FLOAT[2, 2, 2], 8 value(s)
INIT_PX2_1 = [-2.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, 0.0]

# Z: FLOAT[2, 30], 60 value(s)
INIT_Z_2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]

# P: FLOAT[2, 10], 20 value(s)
INIT_P_3 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

# srcpx: FLOAT[2, 2, 2], 8 value(s)
INIT_SRCPX_4 = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, -1.0, 1.2247449159622192]

# eyv: FLOAT[2, 2, 2], 8 value(s)
INIT_EYV_5 = [4.0, -4.0, 0.0, 4.0, 4.0, 0.0, 4.0, 4.898979663848877]

# M23: FLOAT[2, 2], 4 value(s)
INIT_M23_6 = [1.0, 0.3333333432674408, -0.3333333432674408, 0.3333333432674408]

# MD: FLOAT[2, 2], 4 value(s)
INIT_MD_7 = [-0.013888888992369175, -0.25, 0.25, 0.0]

# AC: FLOAT[2, 2], 4 value(s)
INIT_AC_8 = [0.0, -9.0, 20.0, 1.0]

# U4A4: FLOAT[2, 2], 4 value(s)
INIT_U4A4_9 = [-1.0, -2.0, -0.0012248848797753453, -264.57513427734375]

# A: FLOAT[2, 2], 4 value(s)
INIT_A_10 = [1.0, 1.1102086305618286, -1.1045290231704712, -4.285714021534659e-05]


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
        # 0: Einsum inputs=32 outputs=1
        _node(
            'Einsum',
            ['input', 'P', 'AC', 'P', 'P', 'input', 'P', 'AC', 'P', 'P', 'Z', 'px0', 'MD', 'px0', 'MD', 'Z',
             'Z', 'px0', 'MD', 'px0', 'MD', 'Z', 'Z', 'px2', 'px0', 'MD', 'Z', 'Z', 'px0', 'MD', 'Z', 'Z'],
            ['src_row_feat'],
            name='src_row_feat_paireven_norx',
            domain='',
            equation='...fgh,ef,ei,if,if,...pqr,jp,jl,lp,lp,og,aYX,aY,ABa,AB,Ag,Bq,bWZ,bW,CDb,CD,Cg,Dq,Kkc,EFc,EF,Eg,Fq,IJd,IJ,Ig,Jq->...o',
        ),
        # 1: Einsum inputs=32 outputs=1
        _node(
            'Einsum',
            ['input', 'P', 'AC', 'P', 'P', 'input', 'P', 'AC', 'P', 'P', 'Z', 'px0', 'MD', 'px0', 'MD', 'Z',
             'Z', 'px0', 'MD', 'px0', 'MD', 'Z', 'Z', 'px2', 'px0', 'MD', 'Z', 'Z', 'px0', 'MD', 'Z', 'Z'],
            ['src_col_feat'],
            name='src_col_feat_paireven_norx',
            domain='',
            equation='...fgh,ef,ei,if,if,...pqr,jp,jl,lp,lp,oh,aYX,aY,ABa,AB,Ah,Br,bWZ,bW,CDb,CD,Ch,Dr,Kkc,EFc,EF,Eh,Fr,IJd,IJ,Ih,Jr->...o',
        ),
        # 2: Einsum inputs=19 outputs=1
        _node(
            'Einsum',
            ['input', 'px0', 'MD', 'P', 'px2', 'P', 'px0', 'MD', 'P', 'px2', 'M23', 'A', 'U4A4', 'U4A4', 'M23',
             'P', 'px0', 'P', 'Z'],
            ['tgt_row_feat'],
            name='tgt_row_feat_poly5',
            domain='',
            equation='afgh,qrp,qr,qf,Aab,bf,wxv,wx,xf,Ccd,Cc,dd,dd,dd,dd,df,Iij,jf,og->o',
        ),
        # 3: Einsum inputs=19 outputs=1
        _node(
            'Einsum',
            ['input', 'px0', 'MD', 'P', 'px2', 'P', 'px0', 'MD', 'P', 'px2', 'M23', 'A', 'U4A4', 'U4A4', 'M23',
             'P', 'px0', 'P', 'Z'],
            ['tgt_col_feat'],
            name='tgt_col_feat_poly5',
            domain='',
            equation='afgh,qrp,qr,qf,Aab,bf,wxv,wx,xf,Ccd,Cc,dd,dd,dd,dd,df,Iij,jf,oh->o',
        ),
        # 4: Einsum inputs=120 outputs=1
        _node(
            'Einsum',
            ['eyv', 'px0', 'MD', 'Z', 'src_row_feat', 'Z', 'px0', 'MD', 'input', 'srcpx', 'src_row_feat', 'P',
             'AC', 'P', 'P', 'px0', 'px0', 'MD', 'tgt_row_feat', 'Z', 'tgt_row_feat', 'px0', 'MD', 'Z',
             'tgt_row_feat', 'px0', 'MD', 'Z', 'px2', 'px2', 'M23', 'A', 'U4A4', 'U4A4', 'M23', 'px0', 'MD',
             'tgt_row_feat', 'Z', 'eyv', 'MD', 'MD', 'M23', 'A', 'U4A4', 'U4A4', 'M23', 'U4A4', 'Z', 'eyv',
             'px0', 'MD', 'src_col_feat', 'Z', 'srcpx', 'px0', 'MD', 'src_col_feat', 'px0', 'px0', 'MD',
             'tgt_col_feat', 'Z', 'tgt_col_feat', 'px0', 'MD', 'Z', 'tgt_col_feat', 'px0', 'MD', 'Z', 'px2',
             'px2', 'M23', 'A', 'U4A4', 'U4A4', 'M23', 'px0', 'MD', 'tgt_col_feat', 'Z', 'eyv', 'MD', 'MD',
             'M23', 'A', 'U4A4', 'U4A4', 'M23', 'U4A4', 'P', 'srcpx', 'A', 'A', 'A', 'eyv', 'P', 'input', 'P',
             'A', 'A', 'A', 'eyv', 'P', 'P', 'A', 'U4A4', 'U4A4', 'M23', 'srcpx', 'A', 'U4A4', 'U4A4', 'M23',
             'P', 'A', 'U4A4', 'U4A4', 'M23'],
            ['output'],
            name='output',
            domain='',
            equation='pls,yzs,yz,yg,...z,Ag,ABa,AB,...fgh,pla,...B,if,ij,jf,jf,plm,nom,no,o,nd,r,qrp,qr,qd,u,tuv,tu,td,plv,plv,pl,vv,vv,vv,vv,wxv,wx,x,wd,lpk,kl,kl,kl,ll,ll,ll,ll,kp,Ph,GCJ,PQJ,PQ,...Q,Rh,GCZ,RSZ,RS,...S,GCD,EFD,EF,F,Ee,I,HIG,HI,He,L,KLM,KL,Ke,GCM,GCM,GC,MM,MM,MM,MM,NOM,NO,O,Ne,CGk,kC,kC,kC,CC,CC,CC,CC,kG,Tf,TUV,TU,TV,UV,kUT,Ub,...bde,Vc,WX,WY,XY,kXW,Wf,Xb,WW,WW,WW,WW,WXY,XX,XX,XX,XX,Yc,YY,YY,YY,YY->...cde',
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
            _tensor('px0', TensorProto.FLOAT, (2, 2, 2), INIT_PX0_0),
            _tensor('px2', TensorProto.FLOAT, (2, 2, 2), INIT_PX2_1),
            _tensor('Z', TensorProto.FLOAT, (2, 30), INIT_Z_2),
            _tensor('P', TensorProto.FLOAT, (2, 10), INIT_P_3),
            _tensor('srcpx', TensorProto.FLOAT, (2, 2, 2), INIT_SRCPX_4),
            _tensor('eyv', TensorProto.FLOAT, (2, 2, 2), INIT_EYV_5),
            _tensor('M23', TensorProto.FLOAT, (2, 2), INIT_M23_6),
            _tensor('MD', TensorProto.FLOAT, (2, 2), INIT_MD_7),
            _tensor('AC', TensorProto.FLOAT, (2, 2), INIT_AC_8),
            _tensor('U4A4', TensorProto.FLOAT, (2, 2), INIT_U4A4_9),
            _tensor('A', TensorProto.FLOAT, (2, 2), INIT_A_10),
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
