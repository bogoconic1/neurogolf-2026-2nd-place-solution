from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task032'
TASK_NUM = 32
KAGGLE = {'score': 20.751505, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 70
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task032_cubic_core70'
OPSETS = [('', 22)]


# U: FLOAT[10], 10 value(s)
INIT_U_0 = [-4.849198818206787, 1.3292638063430786, 1.3482424020767212, 1.3334676027297974, 1.3313956260681152, 1.3490591049194336,
 1.3291655778884888, 1.3340469598770142, 1.3334640264511108, 1.359586238861084]

# S: FLOAT[30, 2], 60 value(s)
INIT_S_1 = [-2.9961464405059814, -7.925827503204346, -1.353602409362793, 0.5450084805488586, -1.373526930809021,
 0.6018459796905518, -1.3431165218353271, 0.6369239687919617, -1.3288826942443848, 0.6552206873893738,
 -1.612782597541809, 0.8819776177406311, 0.6606907248497009, 2.8139395713806152, 2.316805362701416, -0.8659126162528992,
 0.6664862036705017, 1.4469244480133057, -0.4667506814002991, -0.18356654047966003, 0.07939289510250092,
 -0.3681723475456238, 0.1381622701883316, 0.46798965334892273, 1.0637410879135132, -0.40320849418640137,
 0.01169273816049099, -0.3607155680656433, -1.3542300462722778, -0.25816452503204346, 1.2592356204986572,
 2.2319893836975098, -0.8179077506065369, 0.6037037372589111, -1.152400255203247, 0.1558983474969864, 1.959837794303894,
 -0.6252014636993408, -0.851554274559021, -0.29840895533561707, -0.16610664129257202, 0.06701436638832092,
 0.10750430077314377, -0.40247291326522827, -0.9084125757217407, -0.2659139335155487, 0.04568694904446602,
 1.7985203266143799, -0.10328403860330582, -0.3108225464820862, -0.004129953216761351, -0.03949759528040886,
 0.6471304893493652, 1.038935661315918, 2.983042001724243, 7.698668956756592, -1.1115189790725708, -0.1869572401046753,
 -0.9694851040840149, 0.41493669152259827]


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
        # 0: Einsum inputs=13 outputs=1
        _node(
            'Einsum',
            ['S', 'S', 'S', 'U', 'S', 'input', 'S', 'input', 'U', 'input', 'U', 'S', 'input'],
            ['output'],
            name='cubic70',
            domain='',
            equation='tf,tf,tg,i,pf,bipw,qf,bkqw,k,bohw,o,rg,bjrw->borw',
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
            _tensor('U', TensorProto.FLOAT, (10,), INIT_U_0),
            _tensor('S', TensorProto.FLOAT, (30, 2), INIT_S_1),
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
