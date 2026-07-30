from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task341'
TASK_NUM = 341
KAGGLE = {'score': 20.140188, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 129
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task341_power129'
OPSETS = [('', 20)]


# P: FLOAT[30, 3], 90 value(s)
INIT_P_0 = [-2.1205384731292725, 2.858769655227661, -1.5148004293441772, -14.871099472045898, 19.62162208557129,
 -10.59302806854248, -2.030107021331787, 3.141942262649536, -1.398753046989441, -0.2820129096508026, 0.7674478888511658,
 -0.084509938955307, 0.3708133399486542, -0.3913349211215973, 0.6990329623222351, 0.9091464877128601, -1.72625732421875,
 1.672231674194336, 0.2781254053115845, -0.9930379986763, 1.0781556367874146, -1.7361515760421753, 1.5224076509475708,
 -0.4896470606327057, -11.911861419677734, 15.247036933898926, -7.969372749328613, -2.8939199447631836,
 3.831749439239502, -2.0167925357818604, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# C: FLOAT[3, 3], 9 value(s)
INIT_C_1 = [-1.4826889038085938, -1.3286622762680054, -0.3868616223335266, -0.45638275146484375, 1.8115391731262207,
 2.2077503204345703, 0.9477141499519348, 2.0422329902648926, 2.643460988998413]

# G: FLOAT[2, 3], 6 value(s)
INIT_G_2 = [3.741001605987549, -0.05819046497344971, -0.7196851372718811, 0.24753764271736145, -0.05967758595943451,
 0.007423557806760073]

# X: FLOAT[10, 2], 20 value(s)
INIT_X_3 = [0.5362032055854797, 0.5308411717414856, 0.0483492948114872, 4.834929859498516e-05, 0.051393259316682816,
 0.00010278652189299464, 0.039938971400260925, 0.00011981691204709932, 0.016606682911515236, 6.64267354295589e-05,
 0.012855732813477516, 6.427866173908114e-05, 0.02976006455719471, 0.0001785603817552328, 0.01655721105635166,
 0.00011590048234211281, 1.4877021312713623, 1.4877021312713623, 0.012817472219467163, 0.00010253977961838245]

# K: FLOAT[2, 2], 4 value(s)
INIT_K_4 = [0.0010000000474974513, 579.789794921875, -579.789794921875, 0.0010000000474974513]


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
        # 0: Einsum inputs=29 outputs=1
        _node(
            'Einsum',
            ['X', 'X', 'input', 'P', 'P', 'G', 'C', 'C', 'C', 'P', 'input', 'P', 'P', 'G', 'C', 'C', 'C', 'P',
             'K', 'K', 'X', 'X', 'X', 'K', 'X', 'X', 'K', 'X', 'input'],
            ['output'],
            name='',
            domain='',
            equation='ao,az,baxy,xr,ys,Wq,qs,qt,qt,wt,bamn,nR,mu,Wp,pu,pv,pv,hv,Wf,Wf,cf,df,ck,kS,dS,dT,TV,cV,bdhw->bchw',
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
            _tensor('P', TensorProto.FLOAT, (30, 3), INIT_P_0),
            _tensor('C', TensorProto.FLOAT, (3, 3), INIT_C_1),
            _tensor('G', TensorProto.FLOAT, (2, 3), INIT_G_2),
            _tensor('X', TensorProto.FLOAT, (10, 2), INIT_X_3),
            _tensor('K', TensorProto.FLOAT, (2, 2), INIT_K_4),
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
