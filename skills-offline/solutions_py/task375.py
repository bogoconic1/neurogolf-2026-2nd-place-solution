from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task375'
TASK_NUM = 375
KAGGLE = {'score': 20.905655, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 60
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task375_proj60_selfA'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task375_proj60_selfA'
OPSETS = [('', 12)]


# F: FLOAT[2, 30], 60 value(s)
INIT_F_0 = [2.5, 2.3809523582458496, 2.267573595046997, 2.1595940589904785, 2.0567562580108643, 1.9588154554367065,
 1.865538477897644, 1.7767033576965332, 1.6920983791351318, 1.6115223169326782, 1.534783124923706, 1.4616981744766235,
 1.392093539237976, 1.3258033990859985, 1.2626699209213257, 9.25449275970459, -9.25449275970459, 0.0, -1.0, -1.0,
 1.2599210739135742, -4.671413898468018, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.625, 2.7562499046325684,
 2.8940625190734863, 3.0387656688690186, 3.190703868865967, 3.3502390384674072, 3.5177509784698486, 3.693638563156128,
 3.8783204555511475, 4.07223653793335, 4.275848388671875, 4.489640712738037, 4.714122772216797, 4.9498291015625, -1.0,
 -1.0, 1.2599210739135742, 12.964664459228516, -12.964664459228516, 0.0, 0.0, -9.250320434570312, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0]


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
        # 0: Einsum inputs=49 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'input', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'input', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input', 'F'],
            ['output'],
            name='proj60_selfA',
            domain='',
            equation='aP,aP,bP,ay,cy,noyz,bz,dz,cQ,dQ,dQ,eR,eR,fR,eu,gu,nMuv,fv,kv,gS,kS,kS,lT,lT,pT,lh,nMhw,pw,sw,sU,sU,rU,rh,th,tw,tV,tV,xV,xA,niAB,xB,Yh,Yw,YK,ZK,ZK,ZC,njCD,ZD->nohw',
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
