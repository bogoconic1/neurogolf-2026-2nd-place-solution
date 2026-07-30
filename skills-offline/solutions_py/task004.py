from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task004'
TASK_NUM = 4
KAGGLE = {'score': 20.263802, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 114
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task004_updated'
OPSETS = [('', 18)]


# seed: FLOAT[1, 1, 8, 13], 104 value(s)
INIT_SEED_0 = [26.35215187072754, -8.427106857299805, -8.925812721252441, 0.31595665216445923, -21.692224502563477,
 21.596263885498047, 10.041176795959473, 30.24781608581543, 39.560455322265625, -24.997106552124023, -27.21696662902832,
 -12.100022315979004, -9.449984550476074, -31.746875762939453, -27.685138702392578, -3.9433226585388184,
 21.583938598632812, -11.446854591369629, 11.340764999389648, -6.152232646942139, 1.1193263530731201, 79.14861297607422,
 -69.94209289550781, 8.319522857666016, -1.318474292755127, -0.6603199243545532, -17.745521545410156,
 -10.18048095703125, -0.9648364186286926, -12.14270305633545, 5.210474014282227, 2.74467396736145, -24.308177947998047,
 -39.64280700683594, 135.92079162597656, -119.77198028564453, -3.0653092861175537, -1.8818758726119995,
 40.627315521240234, 11.71461009979248, 3.7645552158355713, 0.1973772495985031, -20.303220748901367, 13.73358154296875,
 -47.549949645996094, 45.1213264465332, -92.87260437011719, 112.30297088623047, -43.15656661987305, 17.35553550720215,
 -0.3536030948162079, -0.4137878715991974, 27.59748077392578, 31.252155303955078, -21.20299530029297, 9.854165077209473,
 24.07350730895996, -18.901796340942383, 3.3658735752105713, 11.088761329650879, 249.948486328125, 300.47344970703125,
 -48.05165481567383, 17.10154151916504, -14.827306747436523, 8.680371284484863, 20.067720413208008, 47.82651138305664,
 21.067102432250977, 9.056690216064453, 21.205629348754883, -4.643794536590576, -1.4008874893188477, 0.6971212029457092,
 49.46126174926758, 36.490638732910156, -23.38241958618164, -41.15190887451172, -71.74560546875, -6.990940093994141,
 -10.793835639953613, -21.639986038208008, -12.399380683898926, 4.145345687866211, -5.321268081665039,
 54.85345458984375, 28.860702514648438, -48.07387161254883, 10.087624549865723, -26.663272857666016, 27.681568145751953,
 14.410452842712402, 9.207857131958008, -53.068572998046875, -0.4993900954723358, -5.995837211608887, 12.49487590789795,
 -7.265066623687744, 21.193218231201172, 11.363996505737305, -8.15147876739502, -5.698616981506348, 1.0182952880859375,
 12.460347175598145]

# bias: FLOAT[10], 10 value(s)
INIT_BIAS_1 = [-299.1158447265625, -327.72454833984375, -322.82696533203125, -341.68707275390625, -335.0789794921875,
 -332.3502502441406, -332.3502502441406, -322.82696533203125, -322.82696533203125, -322.82696533203125]


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
        # 0: ConvTranspose inputs=3 outputs=1
        _node(
            'ConvTranspose',
            ['seed', 'input', 'bias'],
            ['output'],
            name='',
            domain='',
            kernel_shape=[30, 30],
            pads=[4, 8, 3, 4],
            strides=[1, 1],
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
            _tensor('seed', TensorProto.FLOAT, (1, 1, 8, 13), INIT_SEED_0),
            _tensor('bias', TensorProto.FLOAT, (10,), INIT_BIAS_1),
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
