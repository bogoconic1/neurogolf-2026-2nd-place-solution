from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task081'
TASK_NUM = 81
KAGGLE = {'score': 20.617973, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task081_m0p80_rowsumS_dot4'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task081_m0p80_rowsumS_dot4'
OPSETS = [('', 12)]


# S: FLOAT[2, 10], 20 value(s)
INIT_S_0 = [-1.0, 1.9890409708023071, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.00390625, -1.809417724609375, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 2.805511474609375, 0.0]

# U: FLOAT[2, 30], 60 value(s)
INIT_U_1 = [1.0, 0.9249090552330017, 0.7109135985374451, 0.3901517391204834, 0.01079617440700531, -0.37018075585365295,
 -0.6955632567405701, -0.9164848327636719, -0.9997668862342834, -0.9329021573066711, -0.7259324193000793,
 -0.4099406898021698, -0.03238349035382271, 0.3500370979309082, 0.6798888444900513, 0.9076332449913025,
 0.9990676641464233, 0.9404603242874146, 0.7406129837036133, 0.4295386075973511, 0.05395570397377014,
 -0.329730361700058, -0.6638972759246826, -0.8983582854270935, -0.9979026317596436, -0.9475798010826111,
 -0.754948079586029, -0.4489362835884094, -0.07550229132175446, 0.30926987528800964, 0.0, 0.3801884055137634,
 0.7032793760299683, 0.9207505583763123, 0.9999417066574097, 0.9289597272872925, 0.7184648513793945, 0.4000694453716278,
 0.02159108966588974, -0.36012986302375793, -0.6877660155296326, -0.9121121764183044, -0.9994755387306213,
 -0.9367358088493347, -0.7333151698112488, -0.41976410150527954, -0.043172113597393036, 0.3399035334587097,
 0.6719318628311157, 0.9030485153198242, 0.9985433220863342, 0.9440751671791077, 0.7478237748146057, 0.4392634630203247,
 0.06473301351070404, -0.31951919198036194, -0.6557846069335938, -0.8935637474060059, -0.9971456527709961,
 -0.9509742856025696]


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
        # 0: Einsum inputs=26 outputs=1
        _node(
            'Einsum',
            ['U', 'U', 'U', 'U', 'S', 'input', 'S', 'input', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'S',
             'input', 'U', 'U', 'U', 'U', 'input', 'S', 'S', 'S'],
            ['output'],
            name='',
            domain='',
            equation='qs,es,vs,ys,za,nahs,zd,ndts,ph,pt,rh,rt,lh,lt,kh,kt,zb,nbtw,qw,ew,vw,yw,nchw,mc,mo,mj->nohw',
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
            _tensor('S', TensorProto.FLOAT, (2, 10), INIT_S_0),
            _tensor('U', TensorProto.FLOAT, (2, 30), INIT_U_1),
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
