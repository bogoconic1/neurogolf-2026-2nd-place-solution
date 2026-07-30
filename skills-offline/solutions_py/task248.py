from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task248'
TASK_NUM = 248
KAGGLE = {'score': 20.695935, 'date': '2026-07-13'}
MEMORY_BYTES = 8
PARAMS = 66
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task248_expand_simplify_recompress'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task248_factor70_stable'
OPSETS = [('', 18)]


# B: FLOAT[2, 30], 60 value(s)
INIT_B_0 = [0.0, 0.012195262126624584, 0.02439052425324917, 0.03658578544855118, 0.04878104850649834, 0.060976311564445496,
 0.07317157089710236, 0.08536683768033981, 0.09756209701299667, 0.10975735634565353, 0.0, -0.009878162294626236,
 -0.019756324589252472, -0.029634486883878708, -0.039512649178504944, -0.04939081147313118, -0.059268973767757416,
 -0.06914713978767395, -0.07902529835700989, -0.08890345692634583, 1.6201049089431763, -1.6201201677322388,
 -2.600282669067383, 2.600282669067383, 3.137590169906616, -3.137590169906616, -3.113276958465576, 3.113276958465576,
 -1.070662021636963, 1.07066011428833, 0.009878162294626236, 0.009878162294626236, 0.009878162294626236,
 0.009878162294626236, 0.009878162294626236, 0.009878162294626236, 0.009878162294626236, 0.009878162294626236,
 0.009878162294626236, 0.009878162294626236, -0.009878162294626236, -0.009878162294626236, -0.009878162294626236,
 -0.009878162294626236, -0.009878162294626236, -0.009878162294626236, -0.009878162294626236, -0.009878162294626236,
 -0.009878162294626236, -0.009878162294626236, 1.6201201677322388, 1.6201201677322388, -1.5012739896774292,
 -1.5012739896774292, 1.4031730890274048, 1.4031730890274048, -1.270990014076233, -1.270990014076233,
 -1.562441110610962, -1.562441110610962]

# CB: FLOAT[2], 2 value(s)
INIT_CB_1 = [0.5, -1.0]

# K: FLOAT[2, 2], 4 value(s)
INIT_K_2 = [0.0, 4.800000190734863, -4.0, 4.0]


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
        # 0: Einsum inputs=2 outputs=1
        _node(
            'Einsum',
            ['input', 'B'],
            ['F'],
            name='',
            domain='',
            equation='nchw,kw->nk',
        ),
        # 1: Einsum inputs=332 outputs=1
        _node(
            'Einsum',
            ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'input', 'F', 'B', 'B', 'K', 'K', 'K', 'K',
             'K', 'K', 'K', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB',
             'CB', 'F', 'B', 'B', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB',
             'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'F', 'B', 'B', 'K', 'K', 'K', 'K', 'K', 'K', 'CB',
             'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'F', 'B', 'B', 'K', 'K',
             'K', 'K', 'K', 'K', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB',
             'F', 'B', 'B', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB',
             'CB', 'CB', 'CB', 'CB', 'F', 'B', 'B', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'CB', 'CB', 'CB', 'CB',
             'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'F', 'B', 'B', 'K', 'K', 'K', 'K', 'K', 'K', 'K',
             'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'F', 'B',
             'B', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB',
             'CB', 'CB', 'CB', 'CB', 'CB', 'F', 'B', 'B', 'K', 'K', 'K', 'K', 'K', 'K', 'CB', 'CB', 'CB', 'CB',
             'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'F', 'B', 'B', 'K', 'K', 'K', 'K', 'K', 'K', 'CB', 'CB', 'CB',
             'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'F', 'B', 'B', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'CB',
             'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'F', 'B', 'B', 'K', 'K', 'K',
             'K', 'K', 'K', 'K', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'F',
             'B', 'B', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB',
             'CB', 'CB', 'CB', 'F', 'B', 'B', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'CB', 'CB', 'CB', 'CB',
             'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'CB', 'input', 'B', 'CB', 'CB', 'CB'],
            ['output'],
            name='',
            domain='',
            equation='bY,fY,fY,xY,xY,xY,xY,BY,BY,BY,BY,narc,nb,dr,ec,bd,de,eb,bz,dz,ez,zb,b,b,b,b,d,d,d,d,e,e,e,e,z,z,z,nf,gr,hc,fg,gh,hf,fz,gz,hz,zf,f,f,f,f,g,g,g,g,h,h,h,h,z,z,z,ni,jr,kc,ij,jk,ki,iz,jz,kz,i,i,j,j,j,j,k,k,k,k,z,z,z,nl,mr,pc,lm,mp,pl,lz,mz,pz,l,l,m,m,m,m,p,p,p,p,z,z,z,nq,sr,tc,qs,st,tq,qz,sz,tz,zq,q,s,s,s,s,t,t,t,t,z,z,z,nu,vr,wc,uv,vw,wu,uz,vz,wz,zu,u,v,v,v,v,w,w,w,w,z,z,z,nx,yr,Ac,xy,yA,Ax,xz,yz,Az,zx,x,x,x,x,y,y,y,y,A,A,A,A,z,z,z,nB,Cr,Dc,BC,CD,DB,Bz,Cz,Dz,zB,B,B,B,B,C,C,C,C,D,D,D,D,z,z,z,nE,Fr,Gc,EF,FG,GE,Ez,Fz,Gz,F,F,F,F,G,G,G,G,z,z,nH,Ir,Jc,HI,IJ,JH,Hz,Iz,Jz,I,I,I,I,J,J,J,J,z,z,nK,Lr,Mc,KL,LM,MK,Kz,Lz,Mz,zM,K,K,L,L,L,L,M,M,M,M,z,z,nN,Or,Pc,NO,OP,PN,Nz,Oz,Pz,zP,N,N,O,O,O,O,P,P,P,P,z,z,nQ,Rr,Sc,QR,RS,SQ,Qz,Rz,Sz,zQ,zS,Q,R,R,R,R,S,S,S,S,z,z,nT,Ur,Vc,TU,UV,VT,Tz,Uz,Vz,zT,zV,T,U,U,U,U,V,V,V,V,z,z,z,noWX,ZW,Z,Z,Z->norc',
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
            _tensor('B', TensorProto.FLOAT, (2, 30), INIT_B_0),
            _tensor('CB', TensorProto.FLOAT, (2,), INIT_CB_1),
            _tensor('K', TensorProto.FLOAT, (2, 2), INIT_K_2),
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
