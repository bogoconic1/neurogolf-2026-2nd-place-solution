from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task074'
TASK_NUM = 74
KAGGLE = {'score': 20.317869, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 108
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'tiedU'
OPSETS = [('', 20)]


# F: FLOAT[3, 30], 90 value(s)
INIT_F_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -0.8066743612289429, -0.9646572470664978, -0.9951575994491577, -0.894493818283081,
 -0.6756914258003235, -0.3647591173648834, -0.002740500494837761, 0.4003539979457855, 0.7759429812431335,
 0.9824342131614685, 0.9758338332176208, 0.7777902483940125, 0.4986480474472046, 0.15636509656906128,
 -0.20659369230270386, -0.5422664284706116, -0.5422664284706116, -0.20659369230270386, 0.15636509656906128,
 0.4986480474472046, 0.7777902483940125, 0.9758338332176208, 0.9824342131614685, 0.7759429812431335, 0.4003539979457855,
 -0.002740500494837761, -0.3647591173648834, -0.6756914258003235, -0.894493818283081, -0.9951575994491577,
 0.5909962058067322, 0.26350802183151245, -0.0982920229434967, -0.44708025455474854, -0.7371845841407776,
 -0.9311018586158752, -0.9999961853027344, -0.9163604974746704, -0.6308029890060425, -0.18660905957221985,
 0.2185138761997223, 0.6285240054130554, 0.8668045997619629, 0.9876993894577026, 0.9784268736839294, 0.8402066230773926,
 0.8402066230773926, 0.9784268736839294, 0.9876993894577026, 0.8668045997619629, 0.6285240054130554, 0.2185138761997223,
 -0.18660905957221985, -0.6308029890060425, -0.9163604974746704, -0.9999961853027344, -0.9311018586158752,
 -0.7371845841407776, -0.44708025455474854, -0.0982920229434967]

# V: FLOAT[2, 3], 6 value(s)
INIT_V_1 = [1.043203544381921e-11, 3.2325020438150887e-10, 3.2325020438150887e-10, 1.0, 0.0, 0.0]

# R: FLOAT[2], 2 value(s)
INIT_R_2 = [17504894976.0, -0.1826116293668747]

# non9: FLOAT[10], 10 value(s)
INIT_NON9_3 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]


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
        # 0: Einsum inputs=33 outputs=1
        _node(
            'Einsum',
            ['V', 'F', 'V', 'F', 'R', 'V', 'F', 'V', 'F', 'F', 'V', 'F', 'F', 'V', 'F', 'F', 'V', 'F', 'F', 'V',
             'F', 'F', 'V', 'F', 'F', 'V', 'F', 'F', 'V', 'F', 'F', 'input', 'non9'],
            ['output'],
            name='out',
            domain='',
            equation='ap,pz,bq,qz,s,se,ez,ac,ch,cj,ad,dh,dj,bo,ow,oj,br,rw,rj,af,fw,fl,ag,gw,gl,bi,ih,il,bm,mh,ml,nkjl,k->nkhw',
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
            _tensor('F', TensorProto.FLOAT, (3, 30), INIT_F_0),
            _tensor('V', TensorProto.FLOAT, (2, 3), INIT_V_1),
            _tensor('R', TensorProto.FLOAT, (2,), INIT_R_2),
            _tensor('non9', TensorProto.FLOAT, (10,), INIT_NON9_3),
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
