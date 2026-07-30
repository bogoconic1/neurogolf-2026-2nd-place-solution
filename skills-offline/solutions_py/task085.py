from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task085'
TASK_NUM = 85
KAGGLE = {'score': 20.905655, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 60
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 7
PRODUCER_NAME = 'task085-eonly-60'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task085_direct60_eonly'
OPSETS = [('', 12)]


# E: FLOAT[2, 30], 60 value(s)
INIT_E_0 = [-0.46560001373291016, -0.06370000541210175, -0.04540000110864639, -0.056699998676776886, -0.04439999908208847,
 -0.0486999973654747, -0.04360000044107437, -0.09309999644756317, -0.04529999941587448, -0.04659999907016754,
 -0.04149999842047691, -0.05860000103712082, -0.04399999976158142, -0.04990000277757645, -0.034299999475479126,
 -0.05939999967813492, -0.05220000073313713, -0.048100002110004425, -0.05090000107884407, -0.06769999861717224,
 -0.03070000186562538, -0.03199999779462814, -0.05490000173449516, -0.035599999129772186, -0.06109999865293503,
 -0.0494999997317791, -0.042500000447034836, -0.07760000228881836, -0.07029999792575836, -0.149399995803833,
 2.444000005722046, -2.3499999046325684, 2.4579999446868896, -2.7860000133514404, 2.2699999809265137,
 -1.8040000200271606, 2.6559998989105225, -3.364000082015991, 2.25600004196167, -1.8739999532699585, 2.75,
 -2.885999917984009, 2.6519999504089355, -1.8760000467300415, 2.190000057220459, -2.374000072479248, 2.3919999599456787,
 -1.965999960899353, 2.1500000953674316, -2.25, 1.5140000581741333, -1.8259999752044678, 2.315999984741211,
 -2.125999927520752, 2.200000047683716, -2.364000082015991, 2.302000045776367, -2.6019999980926514, 3.049999952316284,
 -2.9719998836517334]


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
        # 0: Einsum inputs=60 outputs=1
        _node(
            'Einsum',
            ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
             'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
             'E', 'E', 'E', 'E', 'input', 'input', 'E', 'E', 'input', 'E', 'E', 'input', 'E', 'E', 'input', 'E',
             'E', 'input', 'E', 'E'],
            ['output'],
            name='direct60_eonly',
            domain='',
            equation='dA,dA,dA,aB,dB,dB,dB,dB,aC,aC,aC,aC,aC,aC,aC,aC,aC,aC,aC,aC,aD,aE,aF,aG,aH,aI,aJ,aK,aL,aM,aN,aO,aP,aQ,aZ,ab,dR,dS,dT,dU,dV,dW,dX,dY,nchw,nciw,ai,ah,nckw,dk,dh,nchj,aj,aw,nchl,fl,fw,nohm,fm,fw->nohw',
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
            _tensor('E', TensorProto.FLOAT, (2, 30), INIT_E_0),
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
