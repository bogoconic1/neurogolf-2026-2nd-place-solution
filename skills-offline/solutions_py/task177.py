from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task177'
TASK_NUM = 177
KAGGLE = {'score': 20.037155, 'date': '2026-07-15'}
MEMORY_BYTES = 21
PARAMS = 122
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task177_bernoulli_scale'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task177_bernoulli_scale'
OPSETS = [('', 15)]


# cmask: FLOAT[10], 10 value(s)
INIT_CMASK_0 = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# keep: FLOAT[30], 30 value(s)
INIT_KEEP_1 = [1.631953832890183e-18, 1.631953832890183e-18, 1.631953832890183e-18, 1.631953832890183e-18, 1.631953832890183e-18,
 1.631953832890183e-18, 1.631953832890183e-18, 1.631953832890183e-18, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# P: FLOAT[2, 30], 60 value(s)
INIT_P_2 = [1.0, 0.9238795042037964, 0.7071067690849304, 0.3826834261417389, 6.123234262925839e-17, -0.3826834261417389,
 -0.7071067690849304, -0.9238795042037964, -1.0, -0.9238795042037964, -0.7071067690849304, -0.3826834261417389,
 -1.8369701465288538e-16, 0.3826834261417389, 0.7071067690849304, 0.9238795042037964, 1.0, 0.9238795042037964,
 0.7071067690849304, 0.19131316244602203, -0.07303304225206375, -0.26611098647117615, -0.453866183757782,
 -0.6202431321144104, -0.7036029696464539, -0.6809907555580139, -0.593591570854187, -0.4847392439842224,
 -0.19672630727291107, 0.32760924100875854, 0.0, 0.3826834261417389, 0.7071067690849304, 0.9238795042037964, 1.0,
 0.9238795042037964, 0.7071067690849304, 0.3826834261417389, 1.2246468525851679e-16, -0.3826834261417389,
 -0.7071067690849304, -0.9238795042037964, -1.0, -0.9238795042037964, -0.7071067690849304, -0.3826834261417389,
 -2.4492937051703357e-16, 0.3826834261417389, 0.7071067690849304, 0.8343033790588379, 0.7682623863220215,
 0.9239644408226013, 0.5459470748901367, 0.16084468364715576, -0.22829829156398773, -0.5444524884223938,
 -0.7528058886528015, -0.8758313059806824, -1.0180304050445557, -0.7327208518981934]

# T1: FLOAT[2, 2, 2], 8 value(s)
INIT_T1_3 = [0.9238795042037964, 0.3826834261417389, -0.3826834261417389, 0.9238795042037964, -0.3826834261417389,
 0.9238795042037964, -0.9238795042037964, -0.3826834261417389]

# T3: FLOAT[2, 2, 2], 8 value(s)
INIT_T3_4 = [0.3826834261417389, 0.9238795042037964, -0.9238795042037964, 0.3826834261417389, -0.9238795042037964,
 0.3826834261417389, -0.3826834261417389, -0.9238795042037964]

# C: FLOAT[2, 2], 4 value(s)
INIT_C_5 = [0.9330329895019531, 0.9330329895019531, 0.9330329895019531, -0.9330329895019531]

# prob: FLOAT[1], 1 value(s)
INIT_PROB_6 = [0.3326700031757355]

# high_scale: FLOAT[1], 1 value(s)
INIT_HIGH_SCALE_7 = [14.463912963867188]


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
        # 0: Einsum inputs=71 outputs=1
        _node(
            'Einsum',
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'T1', 'cmask', 'input', 'P', 'C', 'C', 'C', 'C', 'C',
             'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'P', 'P', 'T3', 'C', 'C', 'C', 'C', 'C', 'P',
             'P', 'P', 'P', 'T3', 'T1', 'P', 'P', 'P', 'P', 'T3', 'P', 'C', 'C', 'C', 'C', 'C', 'P', 'T1', 'P',
             'P', 'input', 'cmask', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['row_phase'],
            name='row_boundary_phase',
            domain='',
            equation='pG,pG,iG,pH,pH,oH,pI,pI,uI,qtu,d,ndsm,ts,tt,te,ee,et,ee,jj,jk,kk,oj,ok,kk,ke,ee,ek,ee,ks,hs,ghi,hh,he,ee,eh,ee,Bs,vs,Es,ys,vwf,xef,qr,gr,xr,jr,EFb,br,BB,BC,CC,DB,DC,Dr,yzA,Ar,ar,ncrl,c,FM,pM,pM,wJ,pJ,pJ,CL,pL,pL,zK,pK,pK->na',
        ),
        # 1: Einsum inputs=76 outputs=1
        _node(
            'Einsum',
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'C', 'C', 'C', 'C', 'C', 'cmask', 'input', 'P', 'P',
             'T3', 'T1', 'P', 'P', 'P', 'P', 'P', 'cmask', 'input', 'P', 'P', 'P', 'P', 'T3', 'C', 'C', 'C',
             'C', 'C', 'T1', 'P', 'C', 'C', 'C', 'C', 'C', 'P', 'P', 'P', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
             'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'T3', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'T1',
             'P', 'P', 'P'],
            ['col_phase'],
            name='col_boundary_phase',
            domain='',
            equation='pH,pH,eH,pI,pI,hI,pJ,pJ,kJ,gg,gh,hh,ig,ih,d,ndms,is,fs,bef,jko,os,qs,Bs,ys,vs,c,nclr,br,jr,gr,tr,qtG,tt,tE,EE,Et,EE,vwx,wr,ww,wE,EE,Ew,EE,zr,Cr,ar,zz,zE,EE,Ez,EE,yy,yz,zz,Ay,Az,CC,CE,EE,EC,EE,BCD,DN,pN,pN,AM,pM,pM,xL,pL,pL,uFG,uK,pK,pK->na',
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['prob'],
            ['sel'],
            name='sel',
            domain='',
            dtype=9,
            seed=239472704.0,
        ),
        # 3: Where inputs=3 outputs=1
        _node(
            'Where',
            ['sel', 'high_scale', 'prob'],
            ['scale'],
            name='scale',
            domain='',
        ),
        # 4: Einsum inputs=116 outputs=1
        _node(
            'Einsum',
            ['scale', 'scale', 'scale', 'row_phase', 'C', 'C', 'C', 'C', 'C', 'row_phase', 'C', 'C', 'C', 'C',
             'C', 'T3', 'P', 'P', 'keep', 'row_phase', 'P', 'T1', 'P', 'row_phase', 'T3', 'T1', 'P', 'P', 'P',
             'P', 'P', 'C', 'C', 'C', 'C', 'C', 'P', 'row_phase', 'C', 'C', 'C', 'C', 'C', 'P', 'T3', 'P',
             'row_phase', 'row_phase', 'C', 'C', 'C', 'C', 'C', 'T1', 'P', 'P', 'C', 'C', 'C', 'C', 'C', 'C',
             'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'input', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'col_phase', 'T3', 'col_phase', 'T3', 'T1', 'col_phase', 'C', 'C', 'C', 'C', 'C',
             'col_phase', 'T3', 'col_phase', 'C', 'C', 'C', 'C', 'C', 'col_phase', 'P', 'P', 'P', 'P', 'P',
             'T1', 'P', 'col_phase', 'T1', 'P', 'keep', 'cmask'],
            ['output'],
            name='half_angle_cover_boundary_phase',
            domain='',
            equation='n,n,n,nu,uu,uv,vv,xu,xv,nb,bb,bW,WW,Wb,WW,abc,ch,vh,h,np,qh,pqt,lh,nk,klX,mWX,mr,ar,tr,xr,fh,dd,de,ee,fd,fe,dr,ne,ee,eW,WW,We,WW,Ar,yzA,zh,ny,ni,ii,iW,WW,Wi,WW,gij,gr,jh,OO,OW,WW,WO,WW,UU,UW,WW,WU,WW,RR,RW,WW,WR,WW,LL,LW,WW,WL,WW,nors,Us,Os,Ls,Rs,Hs,Bs,Es,nT,TUV,nD,BCD,NOP,nN,EE,EF,FF,GE,GF,nG,KLZ,nK,QQ,QR,RR,SQ,SR,nQ,Pw,Vw,Sw,Cw,Fw,MYZ,Mw,nJ,HIJ,Iw,w,o->nohw',
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
            _tensor('cmask', TensorProto.FLOAT, (10,), INIT_CMASK_0),
            _tensor('keep', TensorProto.FLOAT, (30,), INIT_KEEP_1),
            _tensor('P', TensorProto.FLOAT, (2, 30), INIT_P_2),
            _tensor('T1', TensorProto.FLOAT, (2, 2, 2), INIT_T1_3),
            _tensor('T3', TensorProto.FLOAT, (2, 2, 2), INIT_T3_4),
            _tensor('C', TensorProto.FLOAT, (2, 2), INIT_C_5),
            _tensor('prob', TensorProto.FLOAT, (1,), INIT_PROB_6),
            _tensor('high_scale', TensorProto.FLOAT, (1,), INIT_HIGH_SCALE_7),
        ],
        value_info=[
            _vi('sel', TensorProto.BOOL, [1]),
            _vi('scale', TensorProto.FLOAT, [1]),
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
