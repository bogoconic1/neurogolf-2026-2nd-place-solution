from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task033'
TASK_NUM = 33
KAGGLE = {'score': 20.593281, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 82
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'rowscale18'
OPSETS = [('', 18)]


# F: FLOAT[2, 30], 60 value(s)
INIT_F_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.5856571197509766,
 -1.5900578498840332, -1.0334118604660034, -1.4829816818237305, -1.6122124195098877, -1.6168267726898193,
 -0.3301752507686615, -1.5674903392791748, 2.1837568283081055, -1.5856575965881348, -1.5819693803787231,
 -1.6116571426391602, -1.5856586694717407, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0,
 1.0, 0.0, -0.43773528933525085, -0.44249019026756287, 1.2783526182174683, -0.315413236618042, -0.9414324164390564,
 -0.9516205191612244, -1.3154267072677612, -0.447323203086853, 0.8326371312141418, -0.4377349317073822,
 -0.4337766170501709, -0.9403005242347717, -0.4377360939979553]

# H: FLOAT[10, 2], 20 value(s)
INIT_H_1 = [1.0, -1.0, 1.0, 0.05000000074505806, 1.0, 0.15000000596046448, 1.0, 0.25, 1.0, 0.3499999940395355, 1.0,
 0.44999998807907104, 1.0, 0.550000011920929, 1.0, 0.6499999761581421, 1.0, 0.75, 1.0, 0.8199999928474426]

# V: FLOAT[2], 2 value(s)
INIT_V_2 = [0.0015684685204178095, -3.6345338821411133]


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
        # 0: Einsum inputs=100 outputs=1
        _node(
            'Einsum',
            ['V', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'V', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'H', 'H', 'H',
             'H', 'V', 'V', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'H', 'H', 'input', 'F', 'F', 'F', 'F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'H', 'input', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input', 'H', 'F', 'F', 'F', 'F',
             'input', 'F', 'F'],
            ['output'],
            name='output',
            domain='',
            equation='T,TI,wI,wI,wI,wI,rI,rI,G,GJ,RJ,RJ,RJ,RJ,WJ,WJ,dr,dR,cw,cW,da,t,t,tq,tq,tq,tq,tq,tq,tq,tq,tq,tq,tq,tq,tq,tq,tq,tq,tq,tq,tO,tO,aO,aO,aO,aO,aO,ct,cb,bE,bE,tE,sE,sE,sE,sE,Au,Au,gu,gu,es,ep,neuv,gv,gv,gK,Lv,Lv,cB,BF,BF,BF,BF,tF,SF,SF,Cj,Cj,ij,ij,fS,nfjk,fo,ik,ik,iM,Dk,Dk,tU,mU,nhUV,hQ,tV,lV,my,ty,ndyx,tx,lx->ncyx',
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
            _tensor('H', TensorProto.FLOAT, (10, 2), INIT_H_1),
            _tensor('V', TensorProto.FLOAT, (2,), INIT_V_2),
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
