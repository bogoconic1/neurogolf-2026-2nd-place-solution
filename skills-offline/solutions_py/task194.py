from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task194'
TASK_NUM = 194
KAGGLE = {'score': 20.375027, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 102
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task194_factored'
OPSETS = [('', 18)]


# xfac: FLOAT[30, 3], 90 value(s)
INIT_XFAC_0 = [0.9580619931221008, 1.0587260723114014, 0.003932259976863861, 0.9429436326026917, -0.5158240795135498,
 0.9030309915542603, 0.9606309533119202, -0.5332520604133606, -0.9148597121238708, 0.9554917216300964,
 0.5221765041351318, 0.917267918586731, 0.9600591063499451, 0.5273573994636536, -0.9157161116600037, 0.9496946334838867,
 -1.0511934757232666, 0.009444412775337696, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

# qfac: FLOAT[2, 3], 6 value(s)
INIT_QFAC_1 = [0.0021224012598395348, 1.046051025390625, 1.0453009605407715, 0.9541428685188293, 0.0010429715039208531,
 0.0015098484000191092]

# A: FLOAT[2], 2 value(s)
INIT_A_2 = [-3.547244071960449, 3.5728297233581543]

# B: FLOAT[2, 2], 4 value(s)
INIT_B_3 = [-0.38319265842437744, -1.881550908088684, 1.8802838325500488, -2.9079411029815674]


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
        # 0: Einsum inputs=27 outputs=1
        _node(
            'Einsum',
            ['B', 'qfac', 'qfac', 'qfac', 'qfac', 'xfac', 'xfac', 'xfac', 'xfac', 'input', 'xfac', 'xfac',
             'xfac', 'xfac', 'qfac', 'qfac', 'xfac', 'xfac', 'xfac', 'xfac', 'qfac', 'A', 'qfac', 'xfac',
             'xfac', 'xfac', 'xfac'],
            ['output'],
            name='',
            domain='',
            equation='qs,sk,sl,qg,qf,jk,jg,jc,jb,ndij,il,if,ia,ie,rc,re,wl,we,wg,wb,pb,p,pa,hc,ha,hf,hk->ndhw',
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
            _tensor('xfac', TensorProto.FLOAT, (30, 3), INIT_XFAC_0),
            _tensor('qfac', TensorProto.FLOAT, (2, 3), INIT_QFAC_1),
            _tensor('A', TensorProto.FLOAT, (2,), INIT_A_2),
            _tensor('B', TensorProto.FLOAT, (2, 2), INIT_B_3),
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
