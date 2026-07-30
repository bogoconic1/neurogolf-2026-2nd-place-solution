from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task120'
TASK_NUM = 120
KAGGLE = {'score': 20.179718, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 124
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task120_qdiag124'
OPSETS = [('', 18)]


# F: FLOAT[30, 3], 90 value(s)
INIT_F_0 = [1.0, 0.0, 0.9914500117301941, 0.9324722290039062, 0.3612416684627533, 0.9914500117301941, 0.739008903503418,
 0.6736956238746643, 0.9914500117301941, 0.4457383453845978, 0.8951632976531982, 0.9914500117301941,
 0.09226836264133453, 0.9957341551780701, 0.9914500117301941, -0.2736629843711853, 0.9618256688117981,
 0.9914500117301941, -0.602634608745575, 0.7980172038078308, 0.9914500117301941, -0.8502171635627747,
 0.5264321565628052, 0.9914500117301941, -0.9829730987548828, 0.1837495118379593, 0.9914500117301941,
 -0.9829730987548828, -0.1837495118379593, 0.9914500117301941, -0.8502171635627747, -0.5264321565628052,
 0.9914500117301941, -0.602634608745575, -0.7980172038078308, 0.9914500117301941, -0.2736629843711853,
 -0.9618256688117981, 0.9914500117301941, 0.09226836264133453, -0.9957341551780701, 0.9914500117301941,
 0.4457383453845978, -0.8951632976531982, 0.9914500117301941, 0.739008903503418, -0.6736956238746643,
 0.9914500117301941, 1.9324722290039062, 0.6387584805488586, -15.715997695922852, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# U: FLOAT[2, 3], 6 value(s)
INIT_U_1 = [1.0, 1.0, 1.0173218250274658, 0.0, 0.0, -1.7691317796707153]

# Cbin: FLOAT[2, 10], 20 value(s)
INIT_CBIN_2 = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, -2.0, 0.75, -0.75, 0.0, 0.0, 0.0, 0.0, -0.25, 0.0]

# Qbin: FLOAT[2, 2, 2], 8 value(s)
INIT_QBIN_3 = [-1.0473202466964722, 1.339256763458252, -2.945305824279785, 3.7602407932281494, 3.3180761337280273, -4.240030765533447,
 0.7827954292297363, -1.0]


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
        # 0: Einsum inputs=231 outputs=1
        _node(
            'Einsum',
            ['Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'U', 'U', 'Qbin', 'U', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'Qbin', 'F', 'U', 'U', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'F', 'U', 'U', 'Qbin', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'F', 'U', 'U', 'U', 'F', 'F', 'input', 'F', 'F', 'F', 'F', 'U', 'F', 'F',
             'input', 'Cbin', 'Cbin', 'Cbin', 'Qbin', 'F', 'F', 'U', 'F', 'F', 'F', 'U', 'F', 'F', 'F', 'F',
             'Qbin', 'U', 'Qbin', 'F', 'Qbin', 'Qbin', 'F', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Qbin',
             'Qbin', 'Qbin', 'Qbin', 'Qbin', 'Cbin', 'U', 'U', 'Cbin', 'Cbin'],
            ['output'],
            name='output',
            domain='',
            equation='fff,uuu,uuu,uuu,lll,lll,lll,lll,lll,lll,lll,lll,lll,fff,uuu,BBB,BBB,BBB,BBB,BBB,BBB,BBB,uuu,uuu,uuu,uuu,uuu,uuu,uuu,uuu,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,BBB,lll,uuu,ut,BA,fff,fe,fff,uuu,uuu,uuu,uuu,uuu,uuu,dy,lj,ba,rj,ra,rt,re,rA,hA,ha,ht,hj,he,hD,rD,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,WWW,hy,zy,zy,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ,ry,QP,pZ,pE,hE,rE,...crs,sP,sZ,wP,wZ,WV,sV,wV,...chw,qc,qc,kc,kqp,sY,wY,GF,sF,sL,wF,ML,wL,wH,sH,sT,III,IH,III,nT,III,III,wT,MMM,MMM,MMM,MMM,MMM,MMM,MMM,MMM,MMM,MMM,ko,UT,UT,qo,qo->...ohw',
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
            _tensor('F', TensorProto.FLOAT, (30, 3), INIT_F_0),
            _tensor('U', TensorProto.FLOAT, (2, 3), INIT_U_1),
            _tensor('Cbin', TensorProto.FLOAT, (2, 10), INIT_CBIN_2),
            _tensor('Qbin', TensorProto.FLOAT, (2, 2, 2), INIT_QBIN_3),
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
