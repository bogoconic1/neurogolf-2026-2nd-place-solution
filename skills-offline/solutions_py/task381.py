from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task381'
TASK_NUM = 381
KAGGLE = {'score': 20.617973, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task381_p80_exps421_s00minus005'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task381_m0_p80'
OPSETS = [('', 17)]


# S: FLOAT[10, 2], 20 value(s)
INIT_S_0 = [0.4365866184234619, -9.408967162016779e-05, 0.0, 0.0, 1.7002204913296737e-05, 0.609566867351532, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.45995068550109863, -0.0006586420349776745]

# P: FLOAT[2, 30], 60 value(s)
INIT_P_1 = [0.10309632867574692, 0.12275853008031845, 0.12389473617076874, 0.13165521621704102, 0.13303887844085693,
 0.1348252296447754, 0.12925198674201965, 0.1156901866197586, 0.11237136274576187, 0.09637653827667236,
 1.2438756227493286, -1.2438756227493286, -0.4099661707878113, -0.4099661707878113, 0.4931211471557617, 0.0,
 1.4230982065200806, 0.12058744579553604, 0.11804094910621643, 0.06288342922925949, -1.653493046760559,
 0.6933591365814209, 2.184577465057373, 0.2658165693283081, 0.029551370069384575, -1.6178745031356812,
 -1.6525254249572754, 0.0388404056429863, -0.7338690161705017, 0.013123493641614914, -0.028027867898344994,
 0.15260666608810425, 0.2591744065284729, 0.36904555559158325, 0.4689192771911621, 0.5602946877479553,
 0.611920177936554, 0.6685887575149536, 0.7451227307319641, 0.8094028234481812, 0.6805987358093262, 0.6805987358093262,
 0.6309899091720581, -0.6309899091720581, 0.0, -1.3485958576202393, -2.560519218444824, -0.20670828223228455,
 -0.16692028939723969, -0.12602095305919647, 0.42714768648147583, 1.2305314540863037, -0.5172346234321594,
 2.7048470973968506, -0.1285041868686676, 0.41908547282218933, 2.177330255508423, 0.06655201315879822,
 -2.4761252403259277, -0.11463489383459091]


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
        # 0: Einsum inputs=73 outputs=1
        _node(
            'Einsum',
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'S', 'input',
             'P', 'input', 'S', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'input', 'S', 'S'],
            ['output'],
            name='task381_m0_p80_gapfill',
            domain='',
            equation='mf,mf,mf,mf,If,If,Jf,mg,mg,mg,mg,Ag,Ag,Bg,mi,mi,mi,Ki,Ki,mi,Li,Ky,Iy,Ay,Oy,Oj,Oj,mj,mj,mj,mj,Dj,Jx,Lx,Bx,Dx,Ux,bm,nbhx,Vy,nehy,em,Rh,Rk,Rk,mk,mk,mk,mk,Sk,Sh,Ul,ml,ml,ml,ml,Tl,Tl,Vo,Vo,mo,mo,mo,mo,Zo,Tw,Zw,mz,mz,pz,nahw,ap,cp->nchw',
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
            _tensor('S', TensorProto.FLOAT, (10, 2), INIT_S_0),
            _tensor('P', TensorProto.FLOAT, (2, 30), INIT_P_1),
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
