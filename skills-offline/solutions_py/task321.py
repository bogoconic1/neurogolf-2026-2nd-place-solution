from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task321'
TASK_NUM = 321
KAGGLE = {'score': 20.905655, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 60
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'expand-simplify-recompress'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task321_m0_p60'
OPSETS = [('', 18)]


# F: FLOAT[2, 30], 60 value(s)
INIT_F_0 = [-0.25071513652801514, -0.29828205704689026, -0.17609216272830963, 0.07351705431938171, 0.2119019627571106,
 0.2547287344932556, 0.2954949736595154, 0.16937248408794403, -0.04326019063591957, -0.11153675615787506,
 0.24140247702598572, 0.2638324201107025, 0.1525726467370987, -0.010765036568045616, -0.046710625290870667,
 -0.046710625290870667, -0.046710625290870667, -0.046710625290870667, -0.046710625290870667, -0.046710625290870667,
 -0.046710625290870667, -0.046710625290870667, -0.046710625290870667, -0.046710625290870667, -0.046710625290870667,
 -0.046710625290870667, -0.046710625290870667, -0.046710625290870667, -0.046710625290870667, -0.046710625290870667,
 0.1824004054069519, -0.02290787175297737, -0.2505744695663452, -0.2906416654586792, 0.01797519624233246,
 -0.1774604469537735, 0.05073080584406853, 0.2544156014919281, 0.29916462302207947, 0.20344047248363495,
 -0.10858796536922455, 0.08467280119657516, 0.20359492301940918, 0.27634942531585693, -0.046710625290870667,
 -0.046710625290870667, -0.046710625290870667, -0.046710625290870667, -0.046710625290870667, -0.046710625290870667,
 -0.046710625290870667, -0.046710625290870667, -0.046710625290870667, -0.046710625290870667, -0.046710625290870667,
 -0.046710625290870667, -0.046710625290870667, -0.046710625290870667, -0.046710625290870667, -0.046710625290870667]


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
        # 0: Einsum inputs=85 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'input', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input',
             'input', 'input', 'input', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'input', 'input', 'F', 'F'],
            ['output'],
            name='output',
            domain='',
            equation='Bu,Bu,Bu,Du,Eu,bdsu,Ds,Es,Bs,Bs,Bs,as,fs,aw,fw,zs,zw,is,iw,ps,pw,ns,nw,hs,hw,js,jw,ms,mw,gs,gw,ls,lw,ks,kw,qs,qw,Ps,Os,Px,Ox,Gx,Gs,Jx,Js,Lx,Ls,Nx,Ns,Hx,Hs,Mx,Ms,Kx,Ks,Ix,Is,bcrx,borw,bcrs,bctv,Fv,Qs,Rs,Qy,Ry,Us,Uy,Ys,Yy,Ws,Wy,Vs,Vy,Ts,Ty,Xs,Xy,Sy,Ss,Zy,bery,bers,Zs,AC->bors',
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
            _tensor('F', TensorProto.FLOAT, (2, 30), INIT_F_0),
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
