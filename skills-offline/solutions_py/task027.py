from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task027'
TASK_NUM = 27
KAGGLE = {'score': 20.522663, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 88
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 12)]


# C: FLOAT[10, 2], 20 value(s)
INIT_C_0 = [0.0205022431910038, -0.03144867718219757, -0.0008728326065465808, 0.1187092736363411, -0.03063458763062954,
 -0.019846536219120026, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# E: FLOAT[2, 30], 60 value(s)
INIT_E_1 = [0.9969805479049683, 0.9583641886711121, 0.8968619704246521, 0.6826847195625305, 0.3734019994735718,
 -0.022146210074424744, -0.4475783109664917, -0.7030067443847656, -0.8607781529426575, -0.9749360084533691,
 -0.9991263747215271, 0.6564553380012512, 8.037869453430176, -0.17446653544902802, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.022556502372026443, 0.3001011312007904, 0.6452694535255432,
 0.8203626871109009, 0.930064857006073, 1.0294533967971802, 1.0109467506408691, 0.9295293688774109, 0.6989113688468933,
 0.2700623571872711, -0.020425649359822273, -0.5275763869285583, -1.3046597242355347, -0.016079038381576538, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# R: FLOAT[2, 2, 2], 8 value(s)
INIT_R_2 = [-0.9753572344779968, 0.3262025713920593, 0.31123086810112, 0.967043936252594, -1.0222206115722656, 0.15640383958816528,
 0.13127410411834717, 1.0046086311340332]


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
        # 0: Einsum inputs=76 outputs=1
        _node(
            'Einsum',
            ['input', 'C', 'E', 'E', 'R', 'R', 'E', 'E', 'R', 'R', 'C', 'C', 'E', 'E', 'E', 'R', 'E', 'E', 'R',
             'E', 'E', 'R', 'E', 'E', 'R', 'E', 'E', 'R', 'E', 'E', 'R', 'E', 'E', 'R', 'E', 'E', 'R', 'E', 'E',
             'R', 'E', 'E', 'R', 'E', 'E', 'R', 'E', 'E', 'R', 'E', 'E', 'R', 'E', 'E', 'R', 'E', 'E', 'R', 'E',
             'E', 'R', 'E', 'E', 'R', 'E', 'E', 'R', 'E', 'input', 'C', 'input', 'C', 'E', 'E', 'E', 'C'],
            ['output'],
            name='',
            domain='',
            equation='ndxy,dr,mx,jy,kjl,smi,iv,ev,gte,tzf,Xf,Xf,tW,zW,Ap,zAB,Bw,Cp,zCD,Dw,Ep,zEF,Fw,Gp,zGH,Hw,Ip,zIJ,Jw,Kp,zKL,Lw,Mp,zMN,Nw,Op,zOP,Pw,Qp,zQR,Rw,Sp,zST,Tw,Up,zUV,Vw,Yp,zYZ,Zw,Yp,zYZ,Zw,Yp,zYZ,Zw,Yp,zYZ,Zw,Yp,zYZ,Zw,Yp,zYZ,Zw,Yp,zYZ,Zw,nbph,bq,nahw,au,uc,uc,qc,ou->nohw',
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
            _tensor('C', TensorProto.FLOAT, (10, 2), INIT_C_0),
            _tensor('E', TensorProto.FLOAT, (2, 30), INIT_E_1),
            _tensor('R', TensorProto.FLOAT, (2, 2, 2), INIT_R_2),
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
