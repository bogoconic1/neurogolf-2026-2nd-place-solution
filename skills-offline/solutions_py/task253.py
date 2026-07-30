from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task253'
TASK_NUM = 253
KAGGLE = {'score': 20.780492, 'date': '2026-07-10'}
MEMORY_BYTES = 0
PARAMS = 68
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task253_rank2_opt_m0_p84'
OPSETS = [('', 18)]


# B: FLOAT[2, 30], 60 value(s)
INIT_B_0 = [0.001953125, -0.0009765625, 0.00048828125, -0.000244140625, 0.0001220703125, -6.103515625e-05, 3.0517578125e-05,
 -1.52587890625e-05, 7.62939453125e-06, -3.814697265625e-06, 1.9073486328125e-06, -9.5367431640625e-07,
 4.76837158203125e-07, -0.23356719315052032, 0.45327380299568176, -0.7871232032775879, 1.2434141635894775,
 -1.7854201793670654, 2.359548330307007, -2.6307003498077393, 2.6156911849975586, 2.487699031829834, -2.572748899459839,
 2.3442225456237793, -1.8272706270217896, 1.2628592252731323, -0.7943871021270752, 0.4560476243495941,
 -0.2343318909406662, 0.0, -0.000244140625, 0.00048828125, -0.0009765625, 0.001953125, -0.00390625, 0.0078125,
 -0.015625, 0.03125, -0.0625, 0.125, -0.25, 0.5, -1.0, 2.4740686416625977, -2.400660276412964, 2.0844080448150635,
 -1.6463234424591064, 1.1820080280303955, -0.7810501456260681, 0.43540313839912415, -0.21645928919315338,
 0.20586749911308289, -0.4258115887641907, 0.7759770154953003, -1.2097145318984985, 1.6721103191375732,
 -2.1036338806152344, 2.4153504371643066, -2.4821696281433105, 0.0]

# L: FLOAT[2, 2, 2], 8 value(s)
INIT_L_1 = [-631.6795654296875, 1841.6048583984375, -8084.49951171875, -713.9511108398438, -0.009161178022623062,
 0.022075142711400986, -0.010070253163576126, 0.00740167498588562]


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
        # 0: Einsum inputs=47 outputs=1
        _node(
            'Einsum',
            ['input', 'B', 'B', 'input', 'input', 'B', 'L', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'input', 'B', 'input', 'B', 'L', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'input'],
            ['output'],
            name='einsum_rank2_opt',
            domain='',
            equation='nkab,ha,Ar,ncrs,nctd,Bt,AQz,zf,hf,hf,Bf,Bf,Bf,Bf,Pf,Pf,Pf,Pf,Pf,Pf,Pf,Pf,ncvw,Cw,ncey,Dy,CVm,mg,hg,hg,Dg,Dg,Dg,Dg,Ug,Ug,Ug,Ug,Ug,Ug,Ug,Ug,Pi,Qi,Uj,Vj,nxij->ncij',
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
            _tensor('B', TensorProto.FLOAT, (2, 30), INIT_B_0),
            _tensor('L', TensorProto.FLOAT, (2, 2, 2), INIT_L_1),
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
