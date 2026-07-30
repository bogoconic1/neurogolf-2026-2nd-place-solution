from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task371'
TASK_NUM = 371
KAGGLE = {'score': 20.939557, 'date': '2026-07-09'}
MEMORY_BYTES = 8
PARAMS = 50
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 18)]


# T: FLOAT[2, 10], 20 value(s)
INIT_T_0 = [-0.9926691651344299, -0.07407286018133163, 0.08262839913368225, 1.0065760612487793, 0.0737399309873581,
 -0.033987998962402344, -0.033987998962402344, -0.033987998962402344, -0.033987998962402344, -0.03398799151182175,
 -0.0733969435095787, 1.0018107891082764, -1.1175215244293213, 0.07442519813776016, -0.9973080158233643,
 0.421853631734848, 0.421853631734848, 0.421853631734848, 0.421853631734848, 0.42185360193252563]

# Z: FLOAT[30], 30 value(s)
INIT_Z_1 = [9.999999424161284e-19, 1.009999935636435e-18, 1.0199999288567418e-18, 1.0299999220770485e-18, 1.0399999152973552e-18,
 1.0500000119152384e-18, 1.0600000051355451e-18, 1.0699999983558518e-18, 1.0799999915761585e-18, 1.0899999847964652e-18,
 1.0999999780167719e-18, 1.1099999712370786e-18, 1.1199999644573852e-18, 1.129999957677692e-18, 115571924992.0,
 -115571924992.0, 1.0934759127085181e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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
        # 0: Einsum inputs=8 outputs=1
        _node(
            'Einsum',
            ['Z', 'input', 'T', 'T', 'T', 'T', 'input', 'Z'],
            ['prod_moment'],
            name='',
            domain='',
            equation='h,nfgh,kf,kl,de,da,nabc,b->n',
        ),
        # 1: Reciprocal inputs=1 outputs=1
        _node(
            'Reciprocal',
            ['prod_moment'],
            ['prod_recip'],
            name='',
            domain='',
        ),
        # 2: Einsum inputs=36 outputs=1
        _node(
            'Einsum',
            ['Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'prod_recip', 'T', 'T', 'input', 'Z', 'T', 'T',
             'Z', 'T', 'prod_recip', 'T', 'T', 'input', 'Z', 'T', 'T', 'input', 'Z', 'T', 'Z', 'Z', 'T', 'T',
             'T', 'Z', 'Z'],
            ['output'],
            name='',
            domain='',
            equation='u,u,u,p,p,q,q,w,w,w,n,DE,DA,nABC,C,ka,ka,i,zx,n,KL,KF,nFGH,G,lb,lb,nxij,j,zo,r,r,sa,sb,se,v,v->noij',
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
            _tensor('T', TensorProto.FLOAT, (2, 10), INIT_T_0),
            _tensor('Z', TensorProto.FLOAT, (30,), INIT_Z_1),
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
