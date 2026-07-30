from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task270'
TASK_NUM = 270
KAGGLE = {'score': 19.57505, 'date': '2026-07-13'}
MEMORY_BYTES = 97
PARAMS = 130
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task270_m97_p130_kp7'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task270_m97_p130_kp7_graph'
OPSETS = [('', 18)]


# B2: FLOAT[10, 2], 20 value(s)
INIT_B2_0 = [16.0, 0.0, 0.0, 1.0, 1.0000000031710769e-30, -1.0, 16.0, 0.0, -4.9999999724786364e+32, 0.0, 0.0, 0.0,
 4.9999999724786364e+32, 500.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# RB: FLOAT[2, 30], 60 value(s)
INIT_RB_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0044842958450317, 1.0089887380599976, 1.0135133266448975, 1.018058180809021,
 1.0226234197616577, 1.0272091627120972, 1.031815528869629, 1.036442518234253, 1.0410902500152588, 1.0457587242126465,
 1.0504481792449951, 1.0551587343215942, 1.0598903894424438, 1.0646432638168335, 1.0694173574447632, 1.0742130279541016,
 1.0790300369262695, 1.0838687419891357, 1.0887291431427002, 1.0936113595962524, 1.098515510559082, 1.103441596031189,
 1.1083897352218628, 1.113360047340393, 1.1183526515960693, 1.1233676671981812, 1.128405213356018, 1.13346529006958,
 1.1385481357574463]

# E3: FLOAT[10, 2], 20 value(s)
INIT_E3_2 = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# E7: FLOAT[10, 2], 20 value(s)
INIT_E7_3 = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0]

# Kp: FLOAT[2, 2], 4 value(s)
INIT_KP_4 = [0.0020000000949949026, 1.0, -1.0, 0.0]

# M1: FLOAT[2], 2 value(s)
INIT_M1_5 = [1.0, 1.0044842958450317]

# p0: FLOAT[1], 1 value(s)
INIT_P0_6 = [0.44999998807907104]

# p1: FLOAT[1], 1 value(s)
INIT_P1_7 = [0.3100000023841858]

# p2: FLOAT[1], 1 value(s)
INIT_P2_8 = [0.16404031217098236]

# p3: FLOAT[1], 1 value(s)
INIT_P3_9 = [0.5632672905921936]


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
        # 0: Einsum inputs=64 outputs=1
        _node(
            'Einsum',
            ['B2', 'input', 'B2', 'B2', 'E7', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp',
             'Kp', 'Kp', 'Kp', 'Kp', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
             'RB', 'RB', 'RB', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp',
             'Kp', 'RB', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp'],
            ['Vraw'],
            name='Vraw0D',
            domain='',
            equation='cb,nchw,cd,gb,gf,ee,ed,ed,ed,ed,ed,ed,ed,ed,ed,ed,ed,ed,ed,ed,kh,kh,kh,kh,kh,kh,kh,kh,kh,kh,kh,kh,kh,kh,kh,ax,ax,ax,ax,ax,ax,ax,kx,kx,kx,kx,kx,kx,kx,lw,ay,ay,ay,ay,ay,ay,ay,ly,ly,ly,ly,ly,ly,ly->af',
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0'],
            ['up0'],
            name='up0',
            domain='',
            dtype=9,
            seed=9577289.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0'],
            ['up1'],
            name='up1',
            domain='',
            dtype=9,
            seed=241101952.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['down0'],
            name='down0',
            domain='',
            dtype=9,
            seed=1779087104.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['right0'],
            name='right0',
            domain='',
            dtype=9,
            seed=6879713.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['left1'],
            name='left1',
            domain='',
            dtype=9,
            seed=211845184.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['left0'],
            name='left0',
            domain='',
            dtype=9,
            seed=13613418.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['right1'],
            name='right1',
            domain='',
            dtype=9,
            seed=35502252.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0'],
            ['dx'],
            name='dx',
            domain='',
            dtype=9,
            seed=2857688832.0,
        ),
        # 9: Xor inputs=2 outputs=1
        _node(
            'Xor',
            ['right1', 'dx'],
            ['down1'],
            name='down1',
            domain='',
        ),
        # 10: Concat inputs=2 outputs=1
        _node(
            'Concat',
            ['up0', 'up1'],
            ['up'],
            name='up',
            domain='',
            axis=0,
        ),
        # 11: Concat inputs=2 outputs=1
        _node(
            'Concat',
            ['right0', 'right1'],
            ['cpos'],
            name='cpos',
            domain='',
            axis=0,
        ),
        # 12: Concat inputs=2 outputs=1
        _node(
            'Concat',
            ['down0', 'down1'],
            ['rpos'],
            name='rpos',
            domain='',
            axis=0,
        ),
        # 13: Concat inputs=2 outputs=1
        _node(
            'Concat',
            ['left0', 'left1'],
            ['left'],
            name='left',
            domain='',
            axis=0,
        ),
        # 14: Where inputs=3 outputs=1
        _node(
            'Where',
            ['up', 'Vraw', 'Kp'],
            ['V0'],
            name='V0',
            domain='',
        ),
        # 15: Where inputs=3 outputs=1
        _node(
            'Where',
            ['cpos', 'Vraw', 'Kp'],
            ['V1'],
            name='V1',
            domain='',
        ),
        # 16: Where inputs=3 outputs=1
        _node(
            'Where',
            ['rpos', 'Vraw', 'Kp'],
            ['V2'],
            name='V2',
            domain='',
        ),
        # 17: Where inputs=3 outputs=1
        _node(
            'Where',
            ['left', 'Vraw', 'Kp'],
            ['V3'],
            name='V3',
            domain='',
        ),
        # 18: Einsum inputs=561 outputs=1
        _node(
            'Einsum',
            ['Kp', 'V0', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp',
             'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'Kp',
             'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'E3',
             'E3', 'RB', 'V0', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp',
             'Kp', 'Kp', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1',
             'M1', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
             'RB', 'E7', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp',
             'V0', 'Kp', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1',
             'M1', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
             'RB', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'V0', 'E7', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp',
             'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1',
             'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
             'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'V1', 'E3', 'M1', 'Kp', 'RB', 'RB', 'RB', 'RB',
             'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'V1', 'E3', 'M1', 'Kp',
             'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
             'V1', 'E7', 'M1', 'Kp', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
             'RB', 'RB', 'RB', 'RB', 'V1', 'E7', 'M1', 'Kp', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
             'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'V2', 'E3', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1',
             'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'Kp', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
             'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'V2', 'E3', 'M1', 'M1', 'M1', 'M1',
             'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'Kp', 'RB', 'RB', 'RB', 'RB',
             'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'V2', 'E7', 'M1', 'M1',
             'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'Kp', 'RB', 'RB',
             'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'V2', 'E7',
             'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'Kp',
             'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
             'V3', 'E3', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp',
             'Kp', 'M1', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
             'RB', 'RB', 'V3', 'E3', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp',
             'Kp', 'Kp', 'Kp', 'M1', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
             'RB', 'RB', 'RB', 'RB', 'V3', 'E7', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp',
             'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'M1', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
             'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'V3', 'E7', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp',
             'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'Kp', 'M1', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
             'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'B2', 'input', 'B2', 'B2', 'B2', 'Kp', 'E3', 'E7',
             'Kp'],
            ['output'],
            name='output',
            domain='',
            equation='ff,af,aT,aT,aT,aT,aT,aT,aT,aT,aT,aT,aT,aT,aT,aT,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,ba,bh,bh,bh,bh,bh,bh,bh,bh,bh,bh,bh,bh,bh,bh,bh,oa,od,bw,df,dU,dU,dU,dU,dU,dU,dU,dU,dU,dU,dU,dU,dU,dU,de,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,eh,eh,eh,eh,eh,eh,eh,eh,eh,eh,eh,eh,eh,eh,eh,ew,oi,iV,iV,iV,iV,iV,iV,iV,iV,iV,iV,iV,iV,iV,iV,iq,li,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,lh,lh,lh,lh,lh,lh,lh,lh,lh,lh,lh,lh,lh,lh,lh,lw,fq,fq,fq,fq,fq,fq,fq,mq,om,mW,mW,mW,mW,mW,mW,mW,mW,mW,mW,mW,mW,mW,mW,mp,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,ph,ph,ph,ph,ph,ph,ph,ph,ph,ph,ph,ph,ph,ph,ph,pw,rf,or,r,sr,sh,sh,sh,sh,sh,sh,sh,sh,sh,sh,sh,sh,sh,sh,sh,sw,tf,ot,t,tu,uh,uh,uh,uh,uh,uh,uh,uh,uh,uh,uh,uh,uh,uh,uh,uw,vq,ov,v,yv,yh,yh,yh,yh,yh,yh,yh,yh,yh,yh,yh,yh,yh,yh,yh,yw,zq,oz,z,zA,Ah,Ah,Ah,Ah,Ah,Ah,Ah,Ah,Ah,Ah,Ah,Ah,Ah,Ah,Ah,Aw,Bf,oB,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,CB,Ch,Ch,Ch,Ch,Ch,Ch,Ch,Ch,Ch,Ch,Ch,Ch,Ch,Ch,Ch,Cw,Df,oD,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,DE,Eh,Eh,Eh,Eh,Eh,Eh,Eh,Eh,Eh,Eh,Eh,Eh,Eh,Eh,Eh,Ew,Fq,oF,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,GF,Gh,Gh,Gh,Gh,Gh,Gh,Gh,Gh,Gh,Gh,Gh,Gh,Gh,Gh,Gh,Gw,Hq,oH,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,HI,Ih,Ih,Ih,Ih,Ih,Ih,Ih,Ih,Ih,Ih,Ih,Ih,Ih,Ih,Ih,Iw,Jf,oJ,Jx,Jx,Jx,Jx,Jx,Jx,Jx,Jx,Jx,Jx,Jx,Jx,Jx,Jx,KJ,x,Kh,Kh,Kh,Kh,Kh,Kh,Kh,Kh,Kh,Kh,Kh,Kh,Kh,Kh,Kh,Kw,Lf,oL,LX,LX,LX,LX,LX,LX,LX,LX,LX,LX,LX,LX,LX,LX,LM,X,Mh,Mh,Mh,Mh,Mh,Mh,Mh,Mh,Mh,Mh,Mh,Mh,Mh,Mh,Mh,Mw,Nq,oN,NY,NY,NY,NY,NY,NY,NY,NY,NY,NY,NY,NY,NY,NY,ON,Y,Oh,Oh,Oh,Oh,Oh,Oh,Oh,Oh,Oh,Oh,Oh,Oh,Oh,Oh,Oh,Ow,Pq,oP,PZ,PZ,PZ,PZ,PZ,PZ,PZ,PZ,PZ,PZ,PZ,PZ,PZ,PZ,PQ,Z,Qh,Qh,Qh,Qh,Qh,Qh,Qh,Qh,Qh,Qh,Qh,Qh,Qh,Qh,Qh,Qw,oS,nchw,cS,cS,cS,Sj,oj,oj,Rj->nohw',
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
            _tensor('B2', TensorProto.FLOAT, (10, 2), INIT_B2_0),
            _tensor('RB', TensorProto.FLOAT, (2, 30), INIT_RB_1),
            _tensor('E3', TensorProto.FLOAT, (10, 2), INIT_E3_2),
            _tensor('E7', TensorProto.FLOAT, (10, 2), INIT_E7_3),
            _tensor('Kp', TensorProto.FLOAT, (2, 2), INIT_KP_4),
            _tensor('M1', TensorProto.FLOAT, (2,), INIT_M1_5),
            _tensor('p0', TensorProto.FLOAT, (1,), INIT_P0_6),
            _tensor('p1', TensorProto.FLOAT, (1,), INIT_P1_7),
            _tensor('p2', TensorProto.FLOAT, (1,), INIT_P2_8),
            _tensor('p3', TensorProto.FLOAT, (1,), INIT_P3_9),
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
