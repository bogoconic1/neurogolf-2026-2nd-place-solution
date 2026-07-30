from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task203'
TASK_NUM = 203
KAGGLE = {'score': 20.905655, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 60
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task203_R7_self_coefficient_diag_feature_logits'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task203_R7_self_coeff_diag_feature_direct'
OPSETS = [('', 12)]


# U: FLOAT[2, 30], 60 value(s)
INIT_U_0 = [1.6351664066314697, -0.3895193636417389, 3.1339564323425293, 2.5335774421691895, 1.8791730403900146,
 -1.6582282781600952, 1.313880443572998, -0.6813080310821533, -0.04029274731874466, 0.6556321382522583,
 0.12113074958324432, 1.5766173601150513, -2.6237173080444336, 1.3029893636703491, -1.3710235357284546,
 -0.04573396220803261, 3.1863842010498047, 2.8188419342041016, -2.285557985305786, -4.271418571472168,
 7.985050201416016, -4.846174240112305, -6.249652862548828, 3.1812024116516113, -5.396336078643799,
 -0.018986940383911133, -1.040330410003662, 4.081744194030762, -2.3152477741241455, -4.3751935958862305,
 2.53143310546875, 1.3170673847198486, -2.0364038944244385, 3.3928279876708984, 0.08277635276317596,
 -1.2882200479507446, -1.3596277236938477, -0.05193755403161049, -3.954561233520508, -2.330418109893799,
 1.96978759765625, 0.26305875182151794, -2.080143690109253, -2.4269111156463623, -0.938782274723053,
 -1.2186049222946167, -1.5431550741195679, -0.9580351710319519, 2.061959743499756, 0.35197755694389343,
 3.317519187927246, -0.555770754814148, -5.846822261810303, 4.1292033195495605, -2.2785377502441406, 3.3499395847320557,
 2.9650912284851074, 4.202537536621094, 2.0428450107574463, -0.2236272543668747]


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
            ['U', 'U', 'U', 'input', 'U', 'input', 'U', 'input', 'U', 'input', 'U', 'U', 'input'],
            ['output'],
            name='R7_self_coefficient_diag_feature_logits',
            domain='',
            equation='jt,jt,lt,ncee,je,ndff,lf,ncaa,ia,ndbb,ib,iz,nchw->ndhw',
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
            _tensor('U', TensorProto.FLOAT, (2, 30), INIT_U_0),
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
