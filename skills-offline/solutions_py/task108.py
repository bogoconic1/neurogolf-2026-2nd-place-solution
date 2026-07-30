from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task108'
TASK_NUM = 108
KAGGLE = {'score': 20.841117, 'date': '2026-07-11'}
MEMORY_BYTES = 0
PARAMS = 64
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'two_einsum_tied_diag'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task108_m4_p64_tied_diag'
OPSETS = [('', 18)]


# U: FLOAT[2, 30], 60 value(s)
INIT_U_0 = [-0.00892861932516098, -0.046701956540346146, 0.008846703916788101, -0.1540549397468567, 0.17365039885044098,
 0.22337068617343903, -0.17406007647514343, -0.24015282094478607, -0.28598910570144653, -0.28266844153404236,
 0.2857953608036041, -0.28643980622291565, -0.1246674656867981, 0.12490484863519669, 0.12485150992870331,
 0.12482031434774399, -0.26856285333633423, -0.2682421803474426, -0.2684657573699951, 0.26834985613822937,
 -0.20954851806163788, -0.19847580790519714, -0.2017599493265152, -0.21490667760372162, -0.21076808869838715,
 -0.20184074342250824, -0.19886402785778046, -0.20229709148406982, -0.21119338274002075, -0.21028245985507965,
 -0.49124667048454285, 0.6372896432876587, 0.4913056194782257, 0.5340029001235962, 0.2738198935985565,
 0.1782463937997818, -0.27387845516204834, -0.2705945372581482, 0.28703194856643677, 0.3010669946670532,
 -0.2875400483608246, 0.28434860706329346, 0.8283146023750305, -0.8284676671028137, -0.8282575607299805,
 -0.8280413746833801, 0.08234245330095291, 0.0816832184791565, 0.08281467109918594, -0.08166570961475372,
 0.28702056407928467, 0.2861254811286926, 0.28378260135650635, 0.29046693444252014, 0.28165146708488464,
 0.294655978679657, 0.2798234522342682, 0.284027099609375, 0.28169041872024536, 0.28236812353134155]

# G: FLOAT[2, 2], 4 value(s)
INIT_G_1 = [-0.5828308463096619, 0.5178036689758301, -0.033369533717632294, 0.07570555806159973]


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
        # 0: Einsum inputs=97 outputs=1
        _node(
            'Einsum',
            ['G', 'U', 'U', 'G', 'U', 'U', 'U', 'U', 'G', 'U', 'U', 'G', 'U', 'G', 'U', 'U', 'G', 'U', 'U', 'U',
             'U', 'U', 'G', 'U', 'U', 'G', 'U', 'U', 'G', 'U', 'G', 'U', 'U', 'U', 'G', 'U', 'U', 'G', 'U', 'U',
             'G', 'U', 'U', 'G', 'U', 'U', 'G', 'U', 'input', 'U', 'U', 'U', 'G', 'U', 'U', 'G', 'U', 'U', 'G',
             'U', 'U', 'G', 'G', 'U', 'U', 'G', 'U', 'U', 'G', 'U', 'U', 'U', 'U', 'U', 'G', 'U', 'U', 'G', 'U',
             'U', 'G', 'U', 'U', 'G', 'U', 'U', 'G', 'U', 'U', 'G', 'U', 'U', 'G', 'U', 'U', 'G', 'U'],
            ['output'],
            name='task108_tied_diag_p64',
            domain='',
            equation='ab,ar,cr,cd,bx,dx,gx,er,ee,ex,fr,Yf,hr,hh,hx,jr,jj,jx,kr,kl,kx,mr,mn,nx,or,op,px,qr,qq,qx,Yt,tr,ux,vr,vv,vx,zr,zz,zx,Ar,AA,Ax,ir,ii,ix,wr,ww,wx,...xy,Hy,Ty,Cy,BC,Bs,Ds,DE,Ey,Fs,FF,Fy,Gs,ZG,ZS,Ss,Is,II,Iy,Ks,KK,Ky,Ls,LM,Ly,Ns,NO,Oy,Ps,PQ,Qy,Rs,RR,Ry,Us,UU,Uy,Ws,WW,Wy,Xs,XX,Xy,Js,JJ,Jy,Vs,VV,Vy->...rs',
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
            _tensor('U', TensorProto.FLOAT, (2, 30), INIT_U_0),
            _tensor('G', TensorProto.FLOAT, (2, 2), INIT_G_1),
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
