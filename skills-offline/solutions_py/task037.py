from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task037'
TASK_NUM = 37
KAGGLE = {'score': 19.31642, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 294
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 18)]


# H: FLOAT[30, 2], 60 value(s)
INIT_H_0 = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 8.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# L: FLOAT[30, 5], 150 value(s)
INIT_L_1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# U: FLOAT[5, 2, 5], 50 value(s)
INIT_U_2 = [0.48074984550476074, 0.48074984550476074, 0.48074984550476074, 0.48074984550476074, 0.48074984550476074, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.6057068705558777, 0.46399837732315063, 0.10517989099025726, -0.30285343527793884, -0.5691782832145691, 0.0,
 0.389340877532959, 0.596504807472229, 0.5245575308799744, 0.2071639448404312, 0.6057068705558777, 0.10517989099025726,
 -0.5691782832145691, -0.30285343527793884, 0.46399837732315063, 0.0, 0.596504807472229, 0.2071639448404312,
 -0.5245575308799744, -0.389340877532959, 0.6057068705558777, -0.30285343527793884, -0.30285343527793884,
 0.6057068705558777, -0.30285343527793884, 0.0, 0.5245575308799744, -0.5245575308799744, -1.4835539872675715e-16,
 0.5245575308799744, 0.6057068705558777, -0.5691782832145691, 0.46399837732315063, -0.30285343527793884,
 0.10517989099025726, 0.0, 0.2071639448404312, -0.389340877532959, 0.5245575308799744, -0.596504807472229]

# CH: FLOAT[2, 10], 20 value(s)
INIT_CH_3 = [0.004999999888241291, 0.9510565400123596, 0.80901700258255, 0.5877852439880371, 0.30901697278022766,
 -4.371138828673793e-08, -0.30901703238487244, -0.5877852439880371, -0.80901700258255, -0.9510565996170044, 0.0,
 0.30901700258255005, 0.5877852439880371, 0.80901700258255, 0.9510565400123596, 1.0, 0.9510564804077148,
 0.80901700258255, 0.5877852439880371, 0.30901679396629333]

# HP: FLOAT[2, 2], 4 value(s)
INIT_HP_4 = [1.0, 1.0, 1.0, -1.0]

# LP: FLOAT[2, 5], 10 value(s)
INIT_LP_5 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0]


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
        # 0: Einsum inputs=88 outputs=1
        _node(
            'Einsum',
            ['H', 'L', 'HP', 'HP', 'LP', 'HP', 'LP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'U', 'U', 'U', 'HP',
             'HP', 'LP', 'L', 'H', 'input', 'H', 'L', 'LP', 'U', 'U', 'HP', 'HP', 'U', 'LP', 'HP', 'HP', 'HP',
             'LP', 'HP', 'HP', 'HP', 'U', 'U', 'LP', 'L', 'H', 'input', 'H', 'L', 'CH', 'CH', 'U', 'HP', 'LP',
             'HP', 'HP', 'U', 'U', 'U', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'LP', 'LP', 'HP', 'HP', 'LP',
             'L', 'H', 'HP', 'LP', 'HP', 'HP', 'HP', 'H', 'H', 'HP', 'HP', 'HP', 'HP', 'L', 'H', 'L', 'H',
             'input', 'CH', 'CH'],
            ['output'],
            name='',
            domain='',
            equation='ak,av,ii,ik,iv,Lk,Lv,kB,iA,AM,BM,CM,BC,xBv,xAo,xCf,rC,ir,if,Rf,Rr,...cRS,Ss,Sg,Lg,yQv,yQg,Ls,sQ,yQp,Lp,Lj,jQ,Vj,Vp,jP,uP,Vu,wPp,wPn,Vn,Un,Uu,...cTU,Tt,Tm,Kc,Xc,wQq,NQ,Nm,Nt,tG,zGm,zHq,zOo,GW,HW,OW,iO,HO,lH,Vl,Vq,No,Ni,Nl,Nq,eq,el,XE,io,KD,KZ,ZD,bZ,bZ,ZE,YE,YZ,FD,Io,Ii,Jp,Jj,...dIJ,Fh,Yh->...hIJ',
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
            _tensor('H', TensorProto.FLOAT, (30, 2), INIT_H_0),
            _tensor('L', TensorProto.FLOAT, (30, 5), INIT_L_1),
            _tensor('U', TensorProto.FLOAT, (5, 2, 5), INIT_U_2),
            _tensor('CH', TensorProto.FLOAT, (2, 10), INIT_CH_3),
            _tensor('HP', TensorProto.FLOAT, (2, 2), INIT_HP_4),
            _tensor('LP', TensorProto.FLOAT, (2, 5), INIT_LP_5),
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
