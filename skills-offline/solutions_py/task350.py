from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task350'
TASK_NUM = 350
KAGGLE = {'score': 20.569183, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 84
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 18)]


# R: FLOAT[2, 30], 60 value(s)
INIT_R_0 = [-0.125, -0.25, -0.5, -1.0, -2.0, -4.0, -8.0, -16.0, -32.0, -64.0, -128.0, -256.0, -512.0, -1024.0, -2048.0, -4096.0,
 -8192.0, -16384.0, -32768.0, -65536.0, -131072.0, -262144.0, -524288.0, -1048576.0, -2097152.0, -4194304.0, -8388608.0,
 -16777216.0, -33554432.0, -67108864.0, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625,
 0.00048828125, 0.000244140625, 0.0001220703125, 6.103515625e-05, 3.0517578125e-05, 1.52587890625e-05,
 7.62939453125e-06, 3.814697265625e-06, 1.9073486328125e-06, 9.5367431640625e-07, 4.76837158203125e-07,
 2.384185791015625e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 2.9802322387695312e-08, 1.4901161193847656e-08,
 7.450580596923828e-09, 3.725290298461914e-09, 1.862645149230957e-09, 9.313225746154785e-10, 4.656612873077393e-10,
 2.3283064365386963e-10, 1.1641532182693481e-10]

# C: FLOAT[2, 10], 20 value(s)
INIT_C_1 = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# A: FLOAT[2, 2], 4 value(s)
INIT_A_2 = [-1.0, 1.0, 1.0, 0.0]


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
            ['A', 'A', 'A', 'A', 'A', 'R', 'R', 'C', 'input', 'R', 'A', 'R', 'A', 'A', 'R', 'R', 'input', 'C',
             'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
             'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'R', 'A', 'A', 'A', 'R',
             'A', 'R', 'R', 'R', 'A', 'A', 'A', 'A', 'C', 'input', 'R', 'R', 'C', 'input', 'R', 'R', 'A', 'A',
             'R', 'input', 'C', 'A', 'C'],
            ['output'],
            name='output',
            domain='',
            equation='KK,oK,to,Yo,tY,Yj,Yj,tb,...brj,Gr,AG,Lr,LL,Ug,Ui,Ui,...ari,ha,hn,nK,Xn,hX,zK,hz,Nh,Nz,DN,BK,tB,OB,Ot,DO,DC,CK,PC,DP,PQ,QR,RS,SF,FK,MF,SM,JK,uJ,TJ,Tu,MT,MV,Vv,VZ,vZ,ZK,vr,uw,wK,Hw,Hr,uH,Xc,tc,Kc,xK,vx,Ix,vI,ve,...emc,Im,Im,ud,...dlc,El,El,Ey,sW,Wc,...frc,pf,pq,qk->...krc',
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
            _tensor('R', TensorProto.FLOAT, (2, 30), INIT_R_0),
            _tensor('C', TensorProto.FLOAT, (2, 10), INIT_C_1),
            _tensor('A', TensorProto.FLOAT, (2, 2), INIT_A_2),
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
