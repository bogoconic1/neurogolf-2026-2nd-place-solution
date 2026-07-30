from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task034'
TASK_NUM = 34
KAGGLE = {'score': 20.617973, 'date': '2026-07-11'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task034_m80'
OPSETS = [('', 20)]


# P: FLOAT[2, 30], 60 value(s)
INIT_P_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0940107107162476, -1.0611473321914673, -1.6636791229248047,
 1.169938087463379, 2.205246925354004, 1.9770640134811401, -1.90969979763031, -1.9970102310180664, -2.3177285194396973,
 -1.8133533000946045, -1.1544713973999023, -1.0124197006225586, 2.089871644973755, 1.512878656387329,
 -0.019873857498168945, 1.9438374042510986, -1.828129529953003, -1.658042550086975, 1.986758828163147,
 1.5115009546279907, -1.5306329727172852, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -2.015805959701538,
 -2.0044524669647217, 1.4932870864868164, -0.06978168338537216, -0.9772846102714539, -1.9958827495574951,
 -1.07269287109375, 0.8283342123031616, 1.466479778289795, -2.208601236343384, -2.062008857727051, -1.8625117540359497,
 -1.2470430135726929, 2.185194969177246, 2.0534989833831787, 1.6034727096557617, 1.390798568725586, 1.9736888408660889,
 -1.4130288362503052, 2.2290377616882324, 1.7571938037872314]

# fg10: FLOAT[10], 10 value(s)
INIT_FG10_1 = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# object10: FLOAT[10], 10 value(s)
INIT_OBJECT10_2 = [-9.999999682655225e-21, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


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
        # 0: Einsum inputs=94 outputs=1
        _node(
            'Einsum',
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'object10', 'input', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'fg10', 'input', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'input', 'input',
             'object10'],
            ['output'],
            name='output',
            domain='',
            equation='lg,lg,lg,kg,lg,ag,kg,aY,dY,dY,eZ,eZ,bZ,bz,az,dz,dz,ez,dz,hz,ez,ez,iP,kP,A,...APQ,oQ,mQ,ac,ic,ic,jc,jc,jc,jc,oE,oE,bE,pE,pE,pE,pE,bu,mu,mu,nu,nu,nu,nu,dG,rG,rG,rG,rG,qG,qG,lR,rR,jR,TR,dL,TL,TL,TL,TL,sL,sL,qU,sU,B,...BUV,xV,vV,vN,vN,eN,wN,wN,wN,wN,wC,nC,pC,eS,xS,xS,yS,yS,yS,yS,yC,...JRC,...KHW,K->...KRC',
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
            _tensor('P', TensorProto.FLOAT, (2, 30), INIT_P_0),
            _tensor('fg10', TensorProto.FLOAT, (10,), INIT_FG10_1),
            _tensor('object10', TensorProto.FLOAT, (10,), INIT_OBJECT10_2),
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
