from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task229'
TASK_NUM = 229
KAGGLE = {'score': 21.821946, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 24
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task229_m0_p24_rowmoment_q5'
OPSETS = [('', 12)]


# E: FLOAT[2, 10], 20 value(s)
INIT_E_0 = [-1.863241195678711, -1.8661874532699585, 1.3484894037246704, 1.2355930805206299, 1.9216169118881226, 0.654403567314148,
 -0.625267505645752, 0.9536560773849487, 2.2191343307495117, -1.9838885068893433, -0.16233773529529572,
 0.39265379309654236, 0.3757154643535614, 0.6221086978912354, -0.7694128751754761, 0.8184398412704468,
 1.3501770496368408, -0.6301867365837097, -0.6251316070556641, 0.3195164203643799]

# F: FLOAT[2, 2], 4 value(s)
INIT_F_1 = [0.003042608965188265, -1.9104164838790894, 2.046510696411133, 0.786522626876831]


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
        # 0: Einsum inputs=57 outputs=1
        _node(
            'Einsum',
            ['E', 'E', 'E', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'E', 'E', 'E', 'E', 'F', 'F', 'F',
             'F', 'F', 'E', 'F', 'F', 'F', 'F', 'F', 'E', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'E', 'E', 'E',
             'E', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input',
             'input', 'input', 'input', 'input', 'input', 'input'],
            ['output'],
            name='',
            domain='',
            equation='ak,bk,ek,fP,fP,aO,PO,aP,gR,gR,aQ,RQ,aR,Oc,Pj,Qc,Rj,iS,iS,bT,ST,bS,Sj,lV,lV,Ub,Vb,UV,Uj,eW,XW,eX,mZ,mZ,eY,ZY,eZ,Wc,Xd,Yc,Zd,...dno,...dpq,...drs,...dtu,...dvx,...dyz,...dAB,...dCD,...dEF,...dGH,...dIJ,...dIK,...dIL,...dIM,...dIN,...chw->...jhw',
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
            _tensor('E', TensorProto.FLOAT, (2, 10), INIT_E_0),
            _tensor('F', TensorProto.FLOAT, (2, 2), INIT_F_1),
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
