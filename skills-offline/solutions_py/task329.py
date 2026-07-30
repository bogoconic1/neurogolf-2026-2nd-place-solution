from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task329'
TASK_NUM = 329
KAGGLE = {'score': 20.617973, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task329_tryhard_greedy_low_live_v11'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task329_tryhard_greedy_low_live_v11'
OPSETS = [('', 12)]


# CE: FLOAT[2, 10], 20 value(s)
INIT_CE_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.25, -0.25, -2.75, -5.25, -7.75, -10.25, -12.75, -15.25, -17.75,
 -20.25]

# CF: FLOAT[2, 30], 60 value(s)
INIT_CF_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -0.8725347518920898, 1.2683732509613037, -1.6530193090438843,
 1.937333583831787, -1.9419492483139038, 1.635772943496704, -1.3675552606582642, 0.7677977085113525, 0.9709657430648804,
 -1.4455487728118896, 1.7346998453140259, -1.7080368995666504, -1.739150881767273, -1.7597448825836182,
 2.5448315143585205, -2.835479259490967, 2.8598976135253906, -2.657339096069336, 2.2616329193115234,
 -1.7024644613265991, 0.0, 0.10000000149011612, 0.30000001192092896, 0.5, 0.699999988079071, 0.9000000357627869,
 1.100000023841858, 1.3000000715255737, 1.5, 1.7000000476837158, 1.7450695037841797, -1.7757225036621094,
 1.3224154710769653, -0.3874667286872864, -0.38838982582092285, 1.308618426322937, -1.9145773649215698,
 1.535595417022705, -1.9419314861297607, 2.365443468093872, -2.2077999114990234, 1.5527608394622803, 0.948627769947052,
 0.3199536204338074, 0.4626966118812561, -1.546625018119812, 2.5999069213867188, -3.382067918777466, 3.7008538246154785,
 -3.4049289226531982, 0.0]


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
        # 0: Einsum inputs=43 outputs=1
        _node(
            'Einsum',
            ['CF', 'CF', 'CF', 'CF', 'CF', 'CF', 'CF', 'CE', 'CE', 'CE', 'CF', 'CF', 'CF', 'CF', 'CF', 'CF',
             'CF', 'CE', 'CF', 'CF', 'CF', 'CE', 'CF', 'CF', 'CF', 'CF', 'CF', 'CF', 'CF', 'CF', 'input',
             'input', 'CF', 'CF', 'CF', 'CF', 'CF', 'CF', 'CF', 'CF', 'CF', 'CF', 'input'],
            ['output'],
            name='tryhard_greedy_low_live_direct_logits',
            domain='',
            equation='aA,aA,dA,dA,dA,dA,fA,aj,lj,dk,lB,lB,fB,mB,mB,mB,mB,mk,fZ,fZ,hZ,hk,hG,uG,uG,vG,vG,vG,vG,vs,bprs,bjrc,uc,xc,hH,xH,xH,yH,yH,yH,yH,yt,bqrt->bkrc',
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
            _tensor('CE', TensorProto.FLOAT, (2, 10), INIT_CE_0),
            _tensor('CF', TensorProto.FLOAT, (2, 30), INIT_CF_1),
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
