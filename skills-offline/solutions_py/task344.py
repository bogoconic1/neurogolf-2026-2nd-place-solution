from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task344'
TASK_NUM = 344
KAGGLE = {'score': 20.617973, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task344_m0_p80'
OPSETS = [('', 12)]


# F: FLOAT[30, 2], 60 value(s)
INIT_F_0 = [1.0192070007324219, -0.08510343730449677, 1.00180184841156, 0.2464124858379364, 0.8723260760307312, 0.559484601020813,
 0.6332557201385498, 0.8115010857582092, 0.3283542990684509, 0.9968827366828918, -0.0046547348611056805,
 1.0469202995300293, -0.36940351128578186, 0.9693520665168762, -0.5992881059646606, 0.8505493998527527,
 -0.8385969400405884, 0.6216878294944763, -0.8646500110626221, 0.5809482336044312, 0.6945756077766418,
 -1.3687117099761963, -1.1554780006408691, 1.3601393699645996, 1.1580733060836792, -0.9977834820747375,
 1.3510609865188599, -0.9449373483657837, -1.326415777206421, 0.45555123686790466, -1.158557415008545,
 -0.35716769099235535, -0.7877919673919678, -0.7894430160522461, -0.6645415425300598, -1.091853141784668, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# E: FLOAT[10, 2], 20 value(s)
INIT_E_1 = [1.0434170961380005, -0.8285802602767944, -3.3158936500549316, -0.7674711346626282, 1.026549220085144,
 -2.8981478214263916, 3.207512140274048, 0.14802949130535126, 3.3758296966552734, 0.7816096544265747,
 -1.1307649612426758, -0.8580348491668701, -3.4100799560546875, -0.7896867990493774, -3.429363250732422,
 -0.7942338585853577, 1.7734376192092896, -0.33576247096061707, 3.276292562484741, 0.7581285238265991]


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
        # 0: Einsum inputs=96 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input',
             'E', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'input', 'E', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'E', 'E'],
            ['output'],
            name='task344_m0_p80_expand_simplify_recompress_reuse_F',
            domain='',
            equation='hf,hg,if,ig,hk,ik,hl,il,hm,im,hn,in,hp,ip,hq,iq,hr,ir,hu,iu,hv,iv,hx,ix,hy,iy,hz,iz,hA,iA,hB,iB,hC,iC,hD,iD,hE,iE,...dij,de,Ue,Ue,Ue,Ue,Ut,Ut,Us,jF,jG,wF,wG,jH,wH,jI,wI,jJ,wJ,jK,wK,jL,wL,jM,wM,jN,wN,jO,wO,jP,wP,jQ,wQ,jS,wS,jT,wT,jV,wV,jW,wW,jX,wX,jY,jZ,wY,wZ,...chw,cb,Rb,Rb,Rs,Rs,Rs,Rs,Ra,oa,os->...ohw',
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
            _tensor('F', TensorProto.FLOAT, (30, 2), INIT_F_0),
            _tensor('E', TensorProto.FLOAT, (10, 2), INIT_E_1),
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
