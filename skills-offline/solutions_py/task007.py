from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task007'
TASK_NUM = 7
KAGGLE = {'score': 20.905655, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 60
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 9
PRODUCER_NAME = 'task007_v5_monomial_fold'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task007_v5_monomial_fold'
OPSETS = [('', 18)]


# M: FLOAT[2, 30], 60 value(s)
INIT_M_0 = [0.9904199242591858, -0.25436875224113464, -0.6423945426940918, 1.043270230293274, -0.24475187063217163,
 -0.6455225944519043, 1.0234622955322266, -1.2065216302871704, -1.883866548538208, -0.5393114686012268,
 -0.23267482221126556, 0.14288564026355743, 1.8707505464553833, -0.050936467945575714, -0.011201313696801662,
 -0.004091412760317326, -0.006003206130117178, -0.026304682716727257, -0.03479277715086937, -0.02442753128707409,
 -0.012838853523135185, -0.010651424527168274, -0.008445183746516705, -0.026408381760120392, 0.0012793071800842881,
 -0.023660339415073395, -0.027059925720095634, -0.04110129922628403, -0.03927966579794884, 0.01745022088289261,
 0.2376076877117157, 0.796530544757843, -1.0128945112228394, 0.256223201751709, 0.8181037902832031, -1.0173003673553467,
 0.24593205749988556, 1.1340209245681763, 1.6595412492752075, 0.3060050904750824, -1.8812907934188843,
 -0.7315657734870911, 0.3885865807533264, 0.06410270929336548, -0.06894133239984512, -0.097959004342556,
 -0.09678599238395691, -0.0517619363963604, -0.12533533573150635, -0.09787509590387344, -0.06779719144105911,
 -0.015623417682945728, -0.07708258181810379, -0.10415966063737869, -0.11720593273639679, -0.17520654201507568,
 -0.0764281302690506, -0.10861712694168091, 0.06425762921571732, -0.04999144747853279]


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
        # 0: Einsum inputs=38 outputs=1
        _node(
            'Einsum',
            ['M', 'M', 'M', 'input', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M',
             'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'input'],
            ['output'],
            name='',
            domain='',
            equation='Bw,Iw,Mw,nchw,Ah,Gh,Lh,Aa,Ba,Pa,Gb,Ib,Qb,Me,Le,Ue,Kg,Tg,Jg,Qz,Tz,Tz,Uy,Vy,Vy,Pf,Ff,Ef,Vi,Ni,Oi,Ks,Fs,Os,Jr,Nr,Er,ndrs->ncrs',
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
            _tensor('M', TensorProto.FLOAT, (2, 30), INIT_M_0),
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
