from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task171'
TASK_NUM = 171
KAGGLE = {'score': 20.751505, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 70
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task171_p70_bform_core_from_coord'
OPSETS = [('', 12)]


# coord_basis: FLOAT[30, 2], 60 value(s)
INIT_COORD_BASIS_0 = [-2.85791015625, 0.0125501099973917, 1.0098285675048828, 2.063654899597168, 0.9890369772911072, 1.85586678981781,
 0.676042914390564, 1.1083829402923584, 0.9449270963668823, 1.1948357820510864, 1.2885948419570923, 0.8943802118301392,
 1.6116279363632202, 0.285506933927536, 2.001077890396118, 0.07101576030254364, 1.2519819736480713,
 -0.0036417499650269747, -0.7309563159942627, -1.1918424367904663, 2.2830770015716553, -0.4584297835826874,
 -1.0263837575912476, -1.0824027061462402, -0.7267171144485474, -1.045441746711731, -1.0456619262695312,
 -0.9466598033905029, -0.14082273840904236, 0.19559547305107117, -0.052878011018037796, -0.8450883030891418,
 -1.5064765214920044, 1.997470498085022, -1.2434662580490112, -0.15084896981716156, -0.589251697063446,
 -1.4055367708206177, -0.7317584156990051, -1.3559049367904663, -0.029984990134835243, -1.1538437604904175,
 1.1077226400375366, -1.353528618812561, -0.39975160360336304, -1.401287317276001, 0.08658309280872345,
 -1.5105642080307007, -0.5788072943687439, -1.02052640914917, -1.129619836807251, -0.34448644518852234,
 -0.18309776484966278, -0.4638519585132599, 0.5097326040267944, -1.3382844924926758, -0.7904107570648193,
 -1.200256109237671, 0.49706724286079407, -0.5728837251663208]

# color_label: FLOAT[10], 10 value(s)
INIT_COLOR_LABEL_1 = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


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
        # 0: Einsum inputs=17 outputs=1
        _node(
            'Einsum',
            ['input', 'coord_basis', 'coord_basis', 'input', 'coord_basis', 'coord_basis', 'coord_basis',
             'coord_basis', 'coord_basis', 'coord_basis', 'input', 'coord_basis', 'coord_basis', 'coord_basis',
             'coord_basis', 'color_label', 'coord_basis'],
            ['output'],
            name='',
            domain='',
            equation='nqxy,xz,xr,ncuv,ua,tz,tr,ta,yz,ys,ndkl,lb,pz,ps,pb,o,hz->noxy',
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
            _tensor('coord_basis', TensorProto.FLOAT, (30, 2), INIT_COORD_BASIS_0),
            _tensor('color_label', TensorProto.FLOAT, (10,), INIT_COLOR_LABEL_1),
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
