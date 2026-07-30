from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task362'
TASK_NUM = 362
KAGGLE = {'score': 20.872866, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 62
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task362_p64_shared_role'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task362_p64'
OPSETS = [('', 12)]


# H: FLOAT[2, 30], 60 value(s)
INIT_H_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -2.591165542602539, 2.860743522644043, -2.7082512378692627,
 -1.8796114921569824, 2.7594714164733887, -2.934368371963501, 2.8968563079833984, -2.5490500926971436,
 2.6849710941314697, 2.6849710941314697, -2.9644484519958496, -0.0, -0.0, 0.0, 1.327404260635376, 1.327404260635376,
 -1.4655730724334717, -1.7724573612213135, -1.7724573612213135, 1.95695161819458, -1.560507893562317,
 -1.234610676765442, -0.9017659425735474, -0.4912485182285309, -0.13172100484371185, 0.18647457659244537,
 0.5195512771606445, 0.844999372959137, 1.1927170753479004, 1.5761116743087769, -1.7941840887069702, 1.7176190614700317,
 -1.1293123960494995, -0.3196958303451538, -0.28342896699905396, 1.0559967756271362, -1.6287437677383423,
 1.7308608293533325, 0.0, 0.0, -0.0, -2.339901924133301, -2.339901924133301, 2.583461284637451, 1.327404260635376,
 1.327404260635376, -1.4655730724334717, 1.7724573612213135, 1.7724573612213135, -1.95695161819458]

# V: FLOAT[2], 2 value(s)
INIT_V_1 = [0.9711583852767944, -0.7652174234390259]


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
        # 0: Einsum inputs=561 outputs=1
        _node(
            'Einsum',
            ['V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'H', 'H', 'H', 'H',
             'H', 'H', 'H', 'H', 'H', 'H', 'H', 'input', 'input', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'input', 'input',
             'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'input', 'input',
             'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'H', 'H', 'H', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'input', 'input', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'H', 'H', 'H', 'H', 'input', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'H', 'H', 'H', 'H', 'H', 'H',
             'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'input'],
            ['output'],
            name='shared_q_role_p64',
            domain='',
            equation='F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,FR,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,GS,GS,GS,GS,GS,GS,GS,GS,GS,GS,GS,nARS,nApq,ap,aJ,aJ,uJ,uJ,uJ,uJ,tJ,uh,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,HT,nBTU,nBrs,br,IU,IU,IU,IU,IU,IU,IU,IU,IU,IU,IU,tK,bK,bK,vK,vK,vK,vK,vh,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,OV,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,nCVW,nCij,cj,PW,PW,PW,PW,PW,PW,PW,PW,PW,PW,PW,cL,cL,yL,yL,yL,yL,xL,tN,xN,xN,yw,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,QY,nDYZ,nDlm,dm,XZ,XZ,XZ,XZ,XZ,XZ,XZ,XZ,XZ,XZ,XZ,xM,dM,dM,zM,zM,zM,zM,zw,nEhw,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,fg,fg,fg,fg,fg,fg,fg,fg,fg,fg,fg,fg,fg,fg,fg,fg,nkgg->nkhw',
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
            _tensor('H', TensorProto.FLOAT, (2, 30), INIT_H_0),
            _tensor('V', TensorProto.FLOAT, (2,), INIT_V_1),
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
