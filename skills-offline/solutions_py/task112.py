from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task112'
TASK_NUM = 112
KAGGLE = {'score': 19.785064, 'date': '2026-07-09'}
MEMORY_BYTES = 80
PARAMS = 104
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task112_packed184_best'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task112_packed184_best'
OPSETS = [('', 18)]


# coord_basis: FLOAT[2, 30], 60 value(s)
INIT_COORD_BASIS_0 = [0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, -3.625,
 -3.375, -3.125, -2.875, -2.625, -2.375, -2.125, -1.875, -1.625, -1.375, -1.125, -0.875, -0.625, -0.375, -0.125, 0.125,
 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875, 2.125, 2.375, 2.625, 2.875, 3.125, 3.375, 3.625]

# channel_embed: FLOAT[10, 2], 20 value(s)
INIT_CHANNEL_EMBED_1 = [0.125244140625, -1.001953125, 0.0, 0.0, 0.124755859375, 0.998046875, -0.125, 0.001953125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# factor_a: FLOAT[2, 2, 2], 8 value(s)
INIT_FACTOR_A_2 = [1.0, -1.6530612707138062, -3.240000009536743, 10.543892860412598, -39.79166793823242, 128.9250030517578,
 65.77806091308594, 8.0]

# factor_b: FLOAT[2, 2, 2], 8 value(s)
INIT_FACTOR_B_3 = [-1.5564335584640503, 1.5564335584640503, -3.12191104888916, 4.548526763916016, -0.005449454765766859,
 0.04904509335756302, 0.0, 0.0]

# distance_form: FLOAT[2, 2, 2], 8 value(s)
INIT_DISTANCE_FORM_4 = [115200.4375, 0.0, 0.0, 0.0, 0.0, -3413.346435546875, 3413.346435546875, 0.0]


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
        # 0: Einsum inputs=5 outputs=1
        _node(
            'Einsum',
            ['channel_embed', 'channel_embed', 'channel_embed', 'input', 'coord_basis'],
            ['row_anchor'],
            name='',
            domain='',
            equation='dq,cq,cq,nchw,ah->na',
        ),
        # 1: Einsum inputs=5 outputs=1
        _node(
            'Einsum',
            ['channel_embed', 'channel_embed', 'channel_embed', 'input', 'coord_basis'],
            ['col_anchor'],
            name='',
            domain='',
            equation='dq,cq,cq,nchw,aw->na',
        ),
        # 2: Einsum inputs=61 outputs=1
        _node(
            'Einsum',
            ['channel_embed', 'channel_embed', 'factor_a', 'coord_basis', 'input', 'coord_basis',
             'distance_form', 'row_anchor', 'factor_a', 'distance_form', 'coord_basis', 'row_anchor',
             'factor_b', 'coord_basis', 'distance_form', 'row_anchor', 'distance_form', 'coord_basis',
             'row_anchor', 'coord_basis', 'distance_form', 'row_anchor', 'factor_a', 'distance_form',
             'coord_basis', 'row_anchor', 'factor_b', 'coord_basis', 'coord_basis', 'distance_form',
             'distance_form', 'row_anchor', 'row_anchor', 'coord_basis', 'distance_form', 'col_anchor',
             'factor_a', 'distance_form', 'coord_basis', 'col_anchor', 'factor_b', 'coord_basis',
             'distance_form', 'col_anchor', 'distance_form', 'coord_basis', 'col_anchor', 'coord_basis',
             'distance_form', 'col_anchor', 'factor_a', 'factor_b', 'distance_form', 'coord_basis',
             'col_anchor', 'coord_basis', 'coord_basis', 'distance_form', 'distance_form', 'col_anchor',
             'col_anchor'],
            ['orbit_bits'],
            name='',
            domain='',
            equation='cq,co,qqo,rZ,nchw,bh,abe,ne,Ara,adf,dh,nf,Arg,ih,gik,nk,gjl,jh,nl,ph,mpt,nt,Bmr,msu,sh,nu,Brv,xh,yh,vxz,vyE,nz,nE,Gw,FGI,nI,CrF,FHJ,Hw,nJ,CrK,Lw,KLN,nN,KMO,Mw,nO,Qw,PQS,nS,DPr,DrU,PRT,Rw,nT,Vw,Ww,UVX,UWY,nX,nY->nABCD',
        ),
        # 3: Einsum inputs=62 outputs=1
        _node(
            'Einsum',
            ['input', 'channel_embed', 'channel_embed', 'coord_basis', 'distance_form', 'row_anchor',
             'coord_basis', 'distance_form', 'row_anchor', 'coord_basis', 'distance_form', 'row_anchor',
             'coord_basis', 'distance_form', 'row_anchor', 'factor_a', 'factor_b', 'channel_embed',
             'coord_basis', 'distance_form', 'row_anchor', 'coord_basis', 'distance_form', 'row_anchor',
             'factor_a', 'coord_basis', 'distance_form', 'row_anchor', 'factor_b', 'distance_form',
             'coord_basis', 'row_anchor', 'orbit_bits', 'coord_basis', 'distance_form', 'col_anchor',
             'factor_a', 'distance_form', 'coord_basis', 'col_anchor', 'factor_b', 'coord_basis',
             'distance_form', 'col_anchor', 'distance_form', 'coord_basis', 'col_anchor', 'factor_a',
             'factor_b', 'coord_basis', 'distance_form', 'col_anchor', 'distance_form', 'coord_basis',
             'col_anchor', 'coord_basis', 'distance_form', 'col_anchor', 'distance_form', 'coord_basis',
             'col_anchor', 'channel_embed'],
            ['output'],
            name='',
            domain='',
            equation='nchw,cq,cq,bh,abe,ne,dh,adf,nf,ih,gik,nk,jh,gjl,nl,Ara,Arg,Zr,ph,mpt,nt,sh,msu,nu,Bmr,xh,vxz,nz,Brv,vyE,yh,nE,nABCD,Gw,FGI,nI,CrF,FHJ,Hw,nJ,CrK,Lw,KLN,nN,KMO,Mw,nO,DPr,DrU,Qw,PQS,nS,PRT,Rw,nT,Vw,UVX,nX,UWY,Ww,nY,oq->nohw',
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
            _tensor('coord_basis', TensorProto.FLOAT, (2, 30), INIT_COORD_BASIS_0),
            _tensor('channel_embed', TensorProto.FLOAT, (10, 2), INIT_CHANNEL_EMBED_1),
            _tensor('factor_a', TensorProto.FLOAT, (2, 2, 2), INIT_FACTOR_A_2),
            _tensor('factor_b', TensorProto.FLOAT, (2, 2, 2), INIT_FACTOR_B_3),
            _tensor('distance_form', TensorProto.FLOAT, (2, 2, 2), INIT_DISTANCE_FORM_4),
        ],
        value_info=[
            _vi('row_anchor', TensorProto.FLOAT, [1, 2]),
            _vi('col_anchor', TensorProto.FLOAT, [1, 2]),
            _vi('orbit_bits', TensorProto.FLOAT, [1, 2, 2, 2, 2]),
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
