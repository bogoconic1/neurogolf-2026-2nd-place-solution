from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task124'
TASK_NUM = 124
KAGGLE = {'score': 20.456705, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 94
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task124_nan_v3'
OPSETS = [('', 17)]


# seed: FLOAT[1, 1, 6, 14], 84 value(s)
INIT_SEED_0 = [-12.787836074829102, -15.512197494506836, -12.12345027923584, 74.66683959960938, -6.128974914550781,
 -11.786389350891113, -12.99731731414795, 4.794834136962891, 12.738322257995605, -3.1526482105255127, -14.646240234375,
 20.278520584106445, 20.278520584106445, 'nan', -17.233741760253906, 0.5230726599693298, -5.0471343994140625,
 18.306907653808594, 10.281606674194336, -6.612229824066162, 5.964970588684082, -2.6372601985931396,
 -0.45390671491622925, -8.370619773864746, 9.635738372802734, -10.488388061523438, 0.0, 'nan', 7.078672409057617,
 -7.006163120269775, -7.52923583984375, 10.454704284667969, -0.33800244331359863, 12.138314247131348,
 -5.007965087890625, 12.775574684143066, 1.1034280061721802, -12.113516807556152, -26.717803955078125,
 -5.483495712280273, -24.682252883911133, 'nan', -5.26228666305542, 1.813916802406311, -13.892708778381348,
 24.655162811279297, -8.1761474609375, -5.479625225067139, 8.880067825317383, -16.291152954101562, 0.8508101105690002,
 22.15959358215332, 7.78916072845459, -0.518919825553894, 3.2603085041046143, 'nan', -18.09724998474121,
 5.632280349731445, 17.018211364746094, 6.708678722381592, 2.0631330013275146, -3.6056954860687256, -13.646838188171387,
 6.892418384552002, -0.8508101105690002, -7.80460262298584, 2.647779703140259, 6.442449569702148, 17.018211364746094,
 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan']

# bias: FLOAT[10], 10 value(s)
INIT_BIAS_1 = [11.787836074829102, -26.91080093383789, -27.76161003112793, -26.91080093383789, -27.32293701171875,
 -24.498126983642578, -26.91080093383789, -26.91080093383789, -21.278520584106445, -26.91080093383789]


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
            dilations=[1, 1],
            group=1,
            kernel_shape=[30, 30],
            pads=[0, 3, 10, 10],
            strides=[2, 1],
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
            _tensor('seed', TensorProto.FLOAT, (1, 1, 6, 14), INIT_SEED_0),
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
