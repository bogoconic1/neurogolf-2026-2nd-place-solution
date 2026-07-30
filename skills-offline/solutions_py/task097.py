from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task097'
TASK_NUM = 97
KAGGLE = {'score': 20.39483, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 100
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task097_two_einsum_p100'
OPSETS = [('', 13)]


# F: FLOAT[30, 3], 90 value(s)
INIT_F_0 = [0.6503716111183167, 0.7596161961555481, 0.0, 0.6503716111183167, 0.7258685827255249, 0.22390080988407135,
 0.6503716111183167, 0.6276243329048157, 0.4279070496559143, 0.6503716111183167, 0.4736129641532898, 0.5938918590545654,
 0.6503716111183167, 0.27751895785331726, 0.7071067690849304, 0.6503716111183167, 0.05676618963479996,
 0.757492184638977, 0.6503716111183167, -0.16903050243854523, 0.7405710220336914, 0.6503716111183167,
 -0.37980809807777405, 0.6578469276428223, 0.6503716111183167, -0.5568380951881409, 0.5166702270507812,
 0.6503716111183167, -0.6843905448913574, 0.32958510518074036, 0.6503716111183167, -0.7511318922042847,
 0.11321491748094559, 0.6503716111183167, -0.7511318922042847, -0.11321491748094559, 0.6503716111183167,
 -0.6843905448913574, -0.32958510518074036, 0.6503716111183167, -0.5568380951881409, -0.5166702270507812,
 0.6503716111183167, -0.37980809807777405, -0.6578469276428223, 0.6503716111183167, -0.16903050243854523,
 -0.7405710220336914, 0.6503716111183167, 0.05676618963479996, -0.757492184638977, 0.6503716111183167,
 0.27751895785331726, -0.7071067690849304, 0.6503716111183167, 0.4736129641532898, -0.5938918590545654,
 0.6503716111183167, 0.6276243329048157, -0.4279070496559143, 8.449991226196289, -3.039309024810791, 1.4549787044525146,
 -3.541567802429199, 8.134860038757324, 8.097315788269043, 5.291714668273926, -1.757994294166565, 0.780318021774292,
 3.4366445541381836, -8.016650199890137, -7.97141695022583, -8.474747657775879, 3.007251739501953, -1.426803469657898,
 -2.718118190765381, 2.0576767921447754, 1.7373921871185303, 4.260278701782227, -6.535528659820557, -6.622406482696533,
 -1.0273103713989258, -0.6244126558303833, 0.641237199306488, -5.11165189743042, 1.9766578674316406,
 -0.9879729747772217, -4.6279215812683105, 6.48151969909668, 6.5617218017578125]

# E: FLOAT[10], 10 value(s)
INIT_E_1 = [1.0, -4.4721360206604, -4.4721360206604, -4.4721360206604, -4.4721360206604, -4.4721360206604, -4.4721360206604,
 -4.4721360206604, -4.4721360206604, -4.4721360206604]


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
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'input', 'E', 'E', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input', 'E', 'input', 'E'],
            ['output'],
            name='make_output',
            domain='',
            equation='ai,ag,ae,af,ab,hb,he,hf,hg,rb,re,rf,rg,hi,ri,hj,rj,hl,rl,hm,rm,hp,rp,ht,rt,hx,rx,hy,ry,hz,rz,hA,rA,hB,rB,hC,rC,hD,rD,hE,rE,rF,hF,qF,qn,...dhw,d,d,qZ,wZ,wM,sZ,sM,wN,sN,wO,sO,wP,sP,wQ,sQ,wR,sR,wS,sS,wT,sT,wU,sU,wV,sV,wW,sW,wX,sX,wY,sY,wH,sH,wJ,wL,wI,wK,sJ,sL,GH,GJ,GL,GI,GK,sI,sK,...crs,c,...ouv,o->...ors',
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
            _tensor('F', TensorProto.FLOAT, (30, 3), INIT_F_0),
            _tensor('E', TensorProto.FLOAT, (10,), INIT_E_1),
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
