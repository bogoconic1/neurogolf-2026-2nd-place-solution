from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task165'
TASK_NUM = 165
KAGGLE = {'score': 20.522663, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 88
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task165_m0_p90'
OPSETS = [('', 14)]


# B: FLOAT[2, 10], 20 value(s)
INIT_B_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]

# T: FLOAT[2, 2, 2], 8 value(s)
INIT_T_1 = [1.0, 0.0, 9.99994610111476e-41, 0.0, 0.0, -1.0, 1.0, 0.0]

# R: FLOAT[2, 30], 60 value(s)
INIT_R_2 = [1.1875, -0.859375, 0.765625, -0.8359375, 0.953125, -1.03515625, 1.05859375, -1.041015625, 1.01171875, -0.9912109375,
 0.9853515625, -0.98974609375, 0.9970703125, -1.002197265625, 1.003662109375, -1.0025634765625, 1.000732421875,
 -0.99945068359375, 0.99908447265625, -0.999359130859375, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 4.0234064994579266e-21, -1.8634724839594607e-19, 1.0625181290357943e-17, -7.424616477180734e-16, 5.417888360170764e-14,
 -3.765876499528531e-12, 2.4647306418046355e-10, -1.5512341633439064e-08, 9.648501873016357e-07,
 -6.0498714447021484e-05, 0.003849029541015625, -0.2474365234375, 15.953125, -1026.25, 65776.0, -4205056.0, 268632064.0,
 -17170432000.0, 1098504994816.0, -70323647021056.0, 3550792271265792.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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
        # 0: Einsum inputs=66 outputs=1
        _node(
            'Einsum',
            ['T', 'T', 'T', 'R', 'input', 'input', 'R', 'T', 'B', 'R', 'R', 'R', 'T', 'T', 'T', 'T', 'T', 'T',
             'T', 'T', 'T', 'B', 'T', 'B', 'B', 'B', 'T', 'B', 'R', 'R', 'input', 'B', 'input', 'R', 'R', 'T',
             'T', 'T', 'T', 'T', 'B', 'B', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'B', 'B', 'B', 'B',
             'B', 'B', 'T', 'T', 'B', 'B', 'B', 'input', 'R', 'R'],
            ['output'],
            name='',
            domain='',
            equation='DDD,DDD,DDD,DM,...HMA,...HPJ,DP,BFB,FH,QL,QL,nL,QQn,QnE,aGT,aGT,aTG,dGW,dGW,bTW,gTd,kH,gik,ij,gj,Ij,gIK,KH,Dt,at,...jtw,gl,...lvw,bv,Dv,ODO,CNO,dsO,EmO,maq,UH,uH,Rqe,Rqe,dqp,dqp,Sqf,Sqf,Vqf,Vqf,fZU,fzu,Zo,zo,fo,fo,yo,Yo,hxy,hYX,xc,Xc,ec,...crw,Dr,dr->...orw',
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
            _tensor('B', TensorProto.FLOAT, (2, 10), INIT_B_0),
            _tensor('T', TensorProto.FLOAT, (2, 2, 2), INIT_T_1),
            _tensor('R', TensorProto.FLOAT, (2, 30), INIT_R_2),
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
