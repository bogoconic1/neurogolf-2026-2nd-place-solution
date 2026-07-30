from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task160'
TASK_NUM = 160
KAGGLE = {'score': 20.094725, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 135
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'gpt-5.6-sol/swarm-20plus'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task160_m0_p135_Rgate'
OPSETS = [('', 18)]


# H: FLOAT[30, 3], 90 value(s)
INIT_H_0 = [1.0, 1.0, 0.0, 0.9999499917030334, 0.80901700258255, 0.5877852439880371, 0.9999499917030334, 0.30901700258255005,
 0.9510565400123596, 0.9999499917030334, -0.30901700258255005, 0.9510565400123596, 0.9999499917030334,
 -0.80901700258255, 0.5877852439880371, 0.9999499917030334, -1.0, 1.2246468525851679e-16, 0.9999499917030334,
 -0.80901700258255, -0.5877852439880371, 0.9999499917030334, -0.30901700258255005, -0.9510565400123596,
 0.9999499917030334, 0.30901700258255005, -0.9510565400123596, 1.0, 0.80901700258255, -0.5877852439880371, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Uroot: FLOAT[2, 3], 6 value(s)
INIT_UROOT_1 = [-0.80901700258255, 1.0, 1.0, -1.0, 1.0, 1.0]

# O: FLOAT[10, 2], 20 value(s)
INIT_O_2 = [1.5709443092346191, -0.30239585041999817, -1.0680807828903198, 1.158826231956482, -0.9750593900680542,
 -1.7492107152938843, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# R: FLOAT[2, 2, 2, 2], 16 value(s)
INIT_R_3 = [1.4407193702936638e-05, 2.1098403522046283e-05, -2.159529458367615e-06, -3.3624310162849724e-06,
 -1.1748514907594654e-06, 1.0474912414792925e-05, -2.1051491785328835e-06, 5.818542831548257e-06, -205.91876220703125,
 911.1233520507812, 57.32308578491211, -1466.66015625, 33.794464111328125, -2024.5318603515625, 629.720703125,
 356.8716125488281]

# D3: FLOAT[3], 3 value(s)
INIT_D3_4 = [0.30901700258255005, 1.0, 1.0]


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
        # 0: Einsum inputs=119 outputs=1
        _node(
            'Einsum',
            ['R', 'Uroot', 'D3', 'D3', 'H', 'H', 'H', 'H', 'Uroot', 'H', 'H', 'Uroot', 'R', 'R', 'R', 'R', 'H',
             'H', 'H', 'H', 'Uroot', 'input', 'O', 'D3', 'H', 'Uroot', 'O', 'input', 'H', 'H', 'D3', 'H', 'H',
             'H', 'H', 'H', 'H', 'Uroot', 'Uroot', 'Uroot', 'H', 'Uroot', 'D3', 'H', 'O', 'input', 'H', 'H',
             'D3', 'H', 'H', 'H', 'H', 'H', 'H', 'Uroot', 'Uroot', 'Uroot', 'H', 'D3', 'Uroot', 'H', 'O',
             'input', 'H', 'H', 'H', 'D3', 'H', 'H', 'H', 'H', 'Uroot', 'H', 'Uroot', 'H', 'Uroot', 'H',
             'Uroot', 'H', 'H', 'H', 'Uroot', 'H', 'Uroot', 'Uroot', 'H', 'D3', 'H', 'H', 'H', 'H', 'H',
             'Uroot', 'H', 'D3', 'H', 'input', 'O', 'H', 'Uroot', 'R', 'Uroot', 'Uroot', 'H', 'Uroot', 'H',
             'Uroot', 'H', 'H', 'H', 'D3', 'H', 'H', 'H', 'H', 'D3', 'H', 'O'],
            ['output'],
            name='task160_p135_swarm_tie',
            domain='',
            equation='ZZZZ,Za,a,b,rb,ra,db,da,Zc,rc,dc,nc,ZZnZ,ZZnZ,ZZnZ,ZZnZ,rf,df,di,ri,ti,...Cds,Ct,j,rj,Zj,Dt,...Des,ej,ek,k,rk,ep,rp,eq,em,rq,tq,nm,Zm,rm,Zu,u,su,Et,...ErF,Fu,Fx,x,sx,sz,Fz,FA,Fy,sA,tA,Zy,ny,sy,B,ZB,sB,Kt,...KrL,LB,sG,LG,G,sI,LI,LH,LJ,tJ,sJ,nH,sH,ZH,sN,tN,sN,rM,rM,tM,rR,ZR,nR,rP,P,hR,hP,hS,rS,hO,ZO,rO,O,rT,...Qrs,Qv,hT,gT,tglv,nW,ZW,sW,ZU,sU,lY,sY,sV,sX,U,wW,wX,wU,wY,V,wV,ov->...ohw',
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
            _tensor('H', TensorProto.FLOAT, (30, 3), INIT_H_0),
            _tensor('Uroot', TensorProto.FLOAT, (2, 3), INIT_UROOT_1),
            _tensor('O', TensorProto.FLOAT, (10, 2), INIT_O_2),
            _tensor('R', TensorProto.FLOAT, (2, 2, 2, 2), INIT_R_3),
            _tensor('D3', TensorProto.FLOAT, (3,), INIT_D3_4),
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
