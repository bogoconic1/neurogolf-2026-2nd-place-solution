from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task302'
TASK_NUM = 302
KAGGLE = {'score': 20.50019, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 90
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'neurogolf-task302-p90-exact-synthM'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task302_direct_einsum_p90_exact_synthM'
OPSETS = [('', 13)]


# F: FLOAT[30, 2], 60 value(s)
INIT_F_0 = [0.977347195148468, -0.21163778007030487, 0.9991545081138611, 0.04108896851539612, 0.9669428467750549,
 0.2549893260002136, 0.8808116316795349, 0.4734647870063782, 0.7410078644752502, 0.6714948415756226, 0.5629367828369141,
 0.8264987468719482, 0.3535038232803345, 0.9354320168495178, 0.12431278079748154, 0.9922420978546143,
 -0.11788009107112885, 0.993026852607727, -0.3463696539402008, 0.9380970597267151, -0.5408581495285034,
 0.8411126136779785, -0.7360199093818665, 0.6769583821296692, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0]

# Z: FLOAT[3, 10], 30 value(s)
INIT_Z_1 = [-2.7983298301696777, -9.954460144042969, -12.322532653808594, -17.22199249267578, 1.9623827934265137,
 4.642892837524414, 0.8329945802688599, -3.2602832317352295, 10.339473724365234, 16.808265686035156, 2.5206964015960693,
 -20.176666259765625, -22.48430633544922, -3.2553813457489014, 18.700593948364258, 4.336320877075195,
 1.3509488105773926, -0.23191651701927185, -0.17362602055072784, 23.193750381469727, -1.155856966972351,
 11.678616523742676, 2.3691656589508057, 9.732508659362793, -1.248158574104309, 3.608238697052002, -3.6151206493377686,
 11.921354293823242, -4.696723461151123, -8.384361267089844]


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
        # 0: Einsum inputs=80 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input', 'input', 'Z', 'Z', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input', 'Z', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input', 'Z', 'input', 'Z', 'Z',
             'Z', 'Z', 'Z', 'Z', 'Z'],
            ['output'],
            name='',
            domain='',
            equation='iA,cA,iB,cB,iC,cC,iD,cD,iE,cE,iF,cF,iG,cG,iH,cH,jI,cI,jJ,cJ,jK,cK,jL,cL,jM,cM,jN,cN,jO,cO,jP,cP,...kri,...lrj,bk,bl,rQ,sQ,rR,sR,rS,sS,rT,sT,rU,sU,rV,sV,rW,sW,rX,sX,...msc,dm,rY,tY,rZ,tZ,ra,ta,re,te,rf,tf,rg,tg,rp,tp,rq,tq,...ntc,dn,...hrc,bh,dh,bo,do,bu,du,vu->...orc',
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
            _tensor('Z', TensorProto.FLOAT, (3, 10), INIT_Z_1),
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
