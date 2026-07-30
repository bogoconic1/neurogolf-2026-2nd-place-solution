from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task141'
TASK_NUM = 141
KAGGLE = {'score': 20.751505, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 70
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task141_p70_synthsel'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task141_p70'
OPSETS = [('', 20)]


# pow2: FLOAT[2, 30], 60 value(s)
INIT_POW2_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 0.8514472842216492, -2.171597480773926, -0.8503220081329346, 0.5837663412094116, -1.1017543077468872,
 2.044255495071411, 2.368638038635254, -0.9720573425292969, -2.3128113746643066, 0.0, 0.10000000149011612,
 0.20000000298023224, 0.30000001192092896, 0.4000000059604645, 0.5, 0.6000000238418579, 0.699999988079071,
 0.800000011920929, 0.8999999761581421, 1.0, 1.100000023841858, 1.2000000476837158, 1.2999999523162842,
 1.399999976158142, 1.5, 1.600000023841858, 1.7000000476837158, 1.7999999523162842, 1.899999976158142, 2.0,
 -1.868260383605957, 1.0994048118591309, -1.9195382595062256, 2.4425837993621826, -2.4573168754577637,
 -0.5832921266555786, -0.675046443939209, -2.11897349357605, 0.21564674377441406]

# r: FLOAT[10], 10 value(s)
INIT_R_1 = [-9.999999319067318e-35, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05,
 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05]


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
        # 0: Einsum inputs=91 outputs=1
        _node(
            'Einsum',
            ['pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2',
             'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2',
             'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2',
             'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2',
             'pow2', 'pow2', 'input', 'r', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2',
             'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2',
             'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2', 'pow2',
             'pow2', 'pow2', 'pow2', 'pow2', 'input', 'input', 'r'],
            ['output'],
            name='',
            domain='',
            equation='mJ,mJ,zJ,zJ,zJ,zJ,xJ,tH,vH,vH,vH,vH,uH,uH,tM,xM,xM,xM,xM,yM,yM,xI,lI,lI,lI,lI,kI,kI,ue,pe,pe,pe,pe,ae,ae,uf,qf,qf,qf,qf,bf,bf,kh,ah,mh,bh,qR,pR,zR,lR,ncRS,c,DS,DL,DL,DL,DL,yL,CL,CL,BS,yK,BK,BK,BK,BK,AK,AK,jS,sS,Cw,Aw,sg,sg,sg,sg,vg,rg,rg,rw,iw,vG,iG,iG,jG,jG,jG,jG,ndhw,noEF,o->nohw',
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
            _tensor('pow2', TensorProto.FLOAT, (2, 30), INIT_POW2_0),
            _tensor('r', TensorProto.FLOAT, (10,), INIT_R_1),
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
