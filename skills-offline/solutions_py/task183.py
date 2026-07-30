from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task183'
TASK_NUM = 183
KAGGLE = {'score': 19.937405, 'date': '2026-07-13'}
MEMORY_BYTES = 16
PARAMS = 142
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'p167_m2gram'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'p167_m2gram'
OPSETS = [('', 18)]


# F: FLOAT[2, 30], 60 value(s)
INIT_F_0 = [0.125, -0.125, -2.0, -5.0, -3.0, -4.0, -1.0, 1.0, -0.25, -0.625, 26.67424964904785, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.125, 1.0, 2.0, 1.0, 1.0, 0.0, -1.0, 0.125, 0.25,
 72.73693084716797, 8.788911819458008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0]

# A0: FLOAT[2, 3, 2], 12 value(s)
INIT_A0_1 = [-1.0, -1.0, 2.0, 5.0, -0.5, -2.0, 2.0, 5.0, -1.0, -1.0, 2.5, 5.0]

# A2: FLOAT[2, 3, 2], 12 value(s)
INIT_A2_2 = [-32768.0, -81920.0, 8192.0, 16384.0, -16384.0, -40960.0, -16384.0, -49152.0, 4.0, 12.0, -0.20000000298023224,
 -0.20000000298023224]

# D3: FLOAT[2, 2], 4 value(s)
INIT_D3_3 = [1.0, 3.0, 0.0, -0.125]

# qvec: FLOAT[10], 10 value(s)
INIT_QVEC_4 = [-9.999553268746908e-13, 0.0, 2.8575906085848146e-08, 2.8575906085848146e-08, 2.8575906085848146e-08,
 2.8575906085848146e-08, 2.8575906085848146e-08, 2.8575906085848146e-08, 3.500000025931118e-17, 2.8575906085848146e-08]

# support: FLOAT[1, 30], 30 value(s)
INIT_SUPPORT_5 = [0.1250000149011612, 0.10717552900314331, -0.14259576797485352, -1.285191535949707, -1.1425957679748535,
 -2.1425957679748535, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0]

# M0: FLOAT[2, 2], 4 value(s)
INIT_M0_6 = [0.0, 1.0, -1.0, 0.0]

# M1: FLOAT[2, 2], 4 value(s)
INIT_M1_7 = [3.0, 12.0, 3.0, 21.0]

# Tsrc: FLOAT[3, 2], 6 value(s)
INIT_TSRC_8 = [3.0, 4.0, 3.0, 1.0, 6.0, 5.0]


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
        # 0: Multinomial inputs=1 outputs=1
        _node(
            'Multinomial',
            ['support'],
            ['rand_i32'],
            name='',
            domain='',
            dtype=6,
            sample_size=1,
            seed=2582.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['rand_i32'],
            ['size_mode'],
            name='',
            domain='',
            max_gram_length=1,
            max_skip_count=0,
            min_gram_length=1,
            mode='TF',
            ngram_counts=[0],
            ngram_indexes=[2, 1, 0, 0, 0, 2, 1, 2, 2, 2, 2, 1, 0, 1, 0, 1, 0, 0, 2, 2, 2, 0, 0],
            pool_int64s=[1, 4, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29],
        ),
        # 2: Einsum inputs=75 outputs=1
        _node(
            'Einsum',
            ['F', 'M0', 'F', 'F', 'M1', 'F', 'F', 'F', 'F', 'F', 'F', 'M1', 'M0', 'M1', 'F', 'F', 'M1', 'F',
             'qvec', 'input', 'F', 'M1', 'M0', 'M1', 'F', 'F', 'M1', 'F', 'F', 'M0', 'F', 'F', 'M1', 'F', 'F',
             'F', 'F', 'F', 'A0', 'F', 'A2', 'F', 'D3', 'F', 'A0', 'F', 'support', 'A0', 'F', 'A2', 'F', 'D3',
             'F', 'A0', 'F', 'support', 'D3', 'Tsrc', 'size_mode', 'D3', 'Tsrc', 'D3', 'Tsrc', 'F', 'F', 'D3',
             'Tsrc', 'input', 'F', 'F', 'qvec', 'qvec', 'qvec', 'support', 'support'],
            ['output'],
            name='m2gram_p167',
            domain='',
            equation='eh,ef,fr,gh,gi,ir,jh,jW,lW,lr,mh,Km,KV,sV,sr,th,ut,ur,d,ndhw,Cw,YC,YZ,DZ,Dc,Ew,FE,Fc,vw,vx,xc,yw,yz,zc,Aw,AX,BX,Bc,pkG,Gr,pkI,Ir,pJ,Jr,HkL,Lr,nr,qkM,Mc,qkO,Oc,qP,Pc,NkS,Sc,nc,qU,kU,nk,pQ,kQ,pR,kR,Ra,Qa,qT,kT,noab,Tb,Ub,o,o,o,nc,nc->norc',
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
            _tensor('F', TensorProto.FLOAT, (2, 30), INIT_F_0),
            _tensor('A0', TensorProto.FLOAT, (2, 3, 2), INIT_A0_1),
            _tensor('A2', TensorProto.FLOAT, (2, 3, 2), INIT_A2_2),
            _tensor('D3', TensorProto.FLOAT, (2, 2), INIT_D3_3),
            _tensor('qvec', TensorProto.FLOAT, (10,), INIT_QVEC_4),
            _tensor('support', TensorProto.FLOAT, (1, 30), INIT_SUPPORT_5),
            _tensor('M0', TensorProto.FLOAT, (2, 2), INIT_M0_6),
            _tensor('M1', TensorProto.FLOAT, (2, 2), INIT_M1_7),
            _tensor('Tsrc', TensorProto.FLOAT, (3, 2), INIT_TSRC_8),
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
