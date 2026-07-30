from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task313'
TASK_NUM = 313
KAGGLE = {'score': 20.905655, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 60
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 12)]


# P: FLOAT[30, 2], 60 value(s)
INIT_P_0 = [1.1577045917510986, -1.0138174295425415, -1.7392728328704834, -0.7443774938583374, 1.2957706451416016,
 -0.5979819893836975, -1.335291862487793, -1.6777923107147217, 1.1169682741165161, -1.4476537704467773,
 0.9079399704933167, 1.5838546752929688, 0.8862924575805664, -0.7758185863494873, -1.1504379510879517,
 -0.49109557271003723, 0.9517375826835632, -0.43458181619644165, -0.9079578518867493, -1.0896028280258179,
 0.8275036811828613, -1.0812208652496338, 0.6740440130233765, 1.179492473602295, 0.877762496471405, -0.7679365873336792,
 -1.2197681665420532, -0.5233572721481323, 0.9918875694274902, -0.45327192544937134, -1.005335807800293,
 -1.1997442245483398, 0.8424451351165771, -1.0988452434539795, 0.7046582698822021, 1.2315229177474976,
 0.943763256072998, -0.8257433772087097, -1.1008421182632446, -0.47180086374282837, 0.09673811495304108,
 1.4621471166610718, -1.492029070854187, 1.6682144403457642, 1.6172356605529785, -1.7444071769714355,
 -1.5818240642547607, 1.675559163093567, -1.302307367324829, 0.47876909375190735, 1.6082658767700195,
 0.7805771231651306, 1.6989078521728516, 0.6827517747879028, 1.4313994646072388, 1.9127224683761597,
 -1.3947869539260864, -1.9004628658294678, -0.33603477478027344, -1.508553385734558]


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
        # 0: Einsum inputs=187 outputs=1
        _node(
            'Einsum',
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'input', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'input'],
            ['output'],
            name='output',
            domain='',
            equation='Ck,Cb,Cb,Cb,Cb,Cb,Cb,Fk,Fk,Fg,Fg,Fs,Fs,Fs,Fs,Fs,Fs,Fs,Fs,Fs,If,If,If,If,If,If,If,If,If,If,If,If,If,If,If,If,If,If,If,If,If,If,If,If,If,KY,KY,KY,KY,KY,KY,Kk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,Jk,KZ,KZ,KZ,KZ,Ee,Ee,Ee,Ea,Ea,Ea,Ea,Ea,Ea,Ea,Ea,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hf,Hk,Hk,he,nqhw,hf,wx,wz,wp,wo,Ak,Ax,Ax,Ay,Ay,Ay,Ay,Al,Al,Al,Al,Al,Al,Al,Al,Bl,Bz,Bz,Bt,Bt,Bt,Bt,Bm,Bm,Bm,Bm,Bm,Bm,Bm,Bm,Gm,Gp,Gp,Gu,Gu,Gu,Gu,Gv,Gv,Gv,Gv,Gv,Gv,Gv,Gv,Dv,Do,Do,Dj,Dj,Dj,Dj,Di,Di,Di,Di,Di,Di,Di,Di,cy,ct,cu,cj,re,rf,ndrc->nqrc',
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
            _tensor('P', TensorProto.FLOAT, (30, 2), INIT_P_0),
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
