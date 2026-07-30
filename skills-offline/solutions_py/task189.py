from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task189'
TASK_NUM = 189
KAGGLE = {'score': 19.701683, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 200
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task189_p200_direct_einsum'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task189_p200_direct_einsum'
OPSETS = [('', 21)]


# Cch: FLOAT[10, 2], 20 value(s)
INIT_CCH_0 = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0, 9.0]

# Rem: FLOAT[30, 3], 90 value(s)
INIT_REM_1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
 0.0, 0.0, 1.0, -3.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.731095406032396e+26, -8.017743572743322e+25, -3.391822755316575e+26,
 1.731095406032396e+26, 8.017743572743322e+25, 3.391822755316575e+26, -6.670326516152309e+25, -1.2065786558986904e+26,
 -3.518855151575531e+26, 6.670326516152309e+25, 1.2065786558986904e+26, 3.518855151575531e+26, 0.0, 2.0, 4.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Bfeat: FLOAT[30, 2], 60 value(s)
INIT_BFEAT_2 = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 0.0, -12.0, -8.0,
 9.999999887266023e-27, 0.0, -9.999999887266023e-27, 0.0, 0.0, 9.999999887266023e-27, 0.0, -9.999999887266023e-27, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 -0.5, -3.0, -33.603519439697266, -6.0, 34.103519439697266, 9.0]

# EqB: FLOAT[3, 2, 2], 12 value(s)
INIT_EQB_3 = [1.0, -1.5, 0.0, 0.5, 0.0, 1.5, 0.5, -1.0, 0.0, -0.5, 0.0, 0.5]

# MapA: FLOAT[2, 3, 3], 18 value(s)
INIT_MAPA_4 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0]


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
        # 0: Einsum inputs=55 outputs=1
        _node(
            'Einsum',
            ['Rem', 'EqB', 'MapA', 'EqB', 'Bfeat', 'Bfeat', 'Rem', 'MapA', 'Bfeat', 'Bfeat', 'Cch', 'Cch',
             'input', 'Cch', 'Bfeat', 'Bfeat', 'Bfeat', 'Cch', 'Cch', 'Bfeat', 'Bfeat', 'Rem', 'EqB', 'MapA',
             'MapA', 'Rem', 'EqB', 'MapA', 'MapA', 'MapA', 'EqB', 'Bfeat', 'Bfeat', 'Rem', 'input', 'Cch',
             'MapA', 'EqB', 'Bfeat', 'Bfeat', 'Rem', 'EqB', 'Bfeat', 'Bfeat', 'Rem', 'EqB', 'Bfeat', 'Bfeat',
             'Rem', 'Bfeat', 'Rem', 'EqB', 'EqB', 'Cch', 'Cch'],
            ['output'],
            name='',
            domain='',
            equation='Ob,bTw,TdG,GPT,BP,BT,BF,TQF,ih,nz,ah,az,...aAB,al,Rl,Rl,Rl,ap,aj,AU,At,Af,gUt,tkg,tqf,SK,Ktu,vqq,vQQ,uqe,eJL,CJ,CL,Cm,...cCD,cp,wQE,EHI,DI,DH,DM,qWX,rX,rW,rm,QYZ,sY,sZ,sM,Nx,NV,Vpj,Vjy,ox,oy->...ors',
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
            _tensor('Cch', TensorProto.FLOAT, (10, 2), INIT_CCH_0),
            _tensor('Rem', TensorProto.FLOAT, (30, 3), INIT_REM_1),
            _tensor('Bfeat', TensorProto.FLOAT, (30, 2), INIT_BFEAT_2),
            _tensor('EqB', TensorProto.FLOAT, (3, 2, 2), INIT_EQB_3),
            _tensor('MapA', TensorProto.FLOAT, (2, 3, 3), INIT_MAPA_4),
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
