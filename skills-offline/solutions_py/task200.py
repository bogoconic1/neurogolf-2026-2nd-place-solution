from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task200'
TASK_NUM = 200
KAGGLE = {'score': 19.963047, 'date': '2026-07-15'}
MEMORY_BYTES = 8
PARAMS = 146
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'golfer-2-r15'
PRODUCER_VERSION = '15'
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task200_power_blocks_safepow160_r16'
OPSETS = [('', 18)]


# P30: FLOAT[30, 2], 60 value(s)
INIT_P30_0 = [0.957627534866333, 0.957627534866333, 0.9603068828582764, 0.95550537109375, 0.962750256061554, 0.9531468152999878,
 0.9644405245780945, 0.9500460624694824, 0.9670286774635315, 0.9478325843811035, 0.9699639678001404, 0.9459560513496399,
 0.9724746346473694, 0.9436625242233276, 0.9744024276733398, 0.9408054947853088, 0.9763012528419495, 0.9379256367683411,
 0.978699266910553, 0.9355282783508301, 6.773243427276611, 6.773243427276611, -1.001460313796997, 0.0, 0.0,
 0.9977723360061646, -0.02578393928706646, -0.02578393928706646, -0.00437204772606492, 0.0, 0.0, -0.003450175281614065,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0]

# P10: FLOAT[10, 2], 20 value(s)
INIT_P10_1 = [1.4772762060165405, 0.3292087912559509, 0.9458837509155273, 0.5729870796203613, 0.9389953017234802, 0.565970242023468,
 0.9390734434127808, 0.5631872415542603, 0.9418755173683167, 0.5620433688163757, 0.6769148707389832, 0.6583244800567627,
 0.9466737508773804, 0.562082052230835, 0.9532964825630188, 0.5631841421127319, 0.9668737053871155, 0.5683491826057434,
 0.960202693939209, 0.56160569190979]

# R: FLOAT[2], 2 value(s)
INIT_R_2 = [1.0025094747543335, -1.0]

# TQ: FLOAT[2], 2 value(s)
INIT_TQ_3 = [1.0, 1.8995333909988403]

# R_pow16: FLOAT[2], 2 value(s)
INIT_R_POW16_4 = [1.040916085243225, 1.0]

# P30_pow160: FLOAT[30, 2], 60 value(s)
INIT_P30_POW160_5 = [0.0009805280715227127, 0.0009805280715227127, 0.0015332276234403253, 0.0006875453400425613, 0.0023024114780128,
 0.0004629864706657827, 0.0030484015587717295, 0.0002748816623352468, 0.004680546931922436, 0.00018926110351458192,
 0.007601446006447077, 0.00013783307804260403, 0.01149542536586523, 9.347026207251474e-05, 0.015781095251441002,
 5.754071389674209e-05, 0.021548425778746605, 3.523262057569809e-05, 0.03190670907497406, 2.3394142772303894e-05, 'inf',
 'inf', 1.2629878520965576, 0.0, 0.0, 0.6998959183692932, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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
        # 0: Einsum inputs=69 outputs=1
        _node(
            'Einsum',
            ['R', 'P30', 'P30', 'P30', 'R', 'P30', 'P30', 'P30', 'R', 'P30', 'P30', 'P30', 'R', 'P30', 'P30',
             'P30', 'TQ', 'TQ', 'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'TQ', 'P10', 'P10',
             'input', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'R', 'P30',
             'P30', 'P30', 'R', 'P30', 'P30', 'P30', 'R', 'P30', 'P30', 'P30', 'R', 'P30', 'P30', 'P30', 'R'],
            ['V'],
            name='r15_einsum_0',
            domain='',
            equation='G,FG,FE,Fu,A,zA,zu,zy,x,wx,wu,wv,p,op,ou,om,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,cE,ca,nchs,sy,sv,sm,sq,sj,sf,sB,sb,gf,gu,gi,i,rq,ru,rt,t,kj,ku,kl,l,db,du,de,e,CB,Cu,CD,D->nu',
        ),
        # 1: Einsum inputs=2187 outputs=1
        _node(
            'Einsum',
            ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'R',
             'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'P10', 'P10', 'V', 'R',
             'P10', 'V', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'TQ', 'P10', 'V', 'R_pow16', 'R_pow16', 'R', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30',
             'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'V', 'P30', 'P30',
             'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
             'R', 'R', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30',
             'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
             'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30',
             'P30', 'P10', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'TQ', 'V', 'P30', 'P30',
             'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
             'R', 'R', 'R', 'R', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
             'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'TQ',
             'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30',
             'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30',
             'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30',
             'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30',
             'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
             'R', 'R', 'R', 'R', 'TQ', 'V', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
             'R', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'P30', 'P30',
             'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P30_pow160', 'P30_pow160',
             'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P30_pow160', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30',
             'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P30_pow160', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P30_pow160', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'R', 'R', 'R',
             'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30',
             'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P30_pow160', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P30_pow160', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R',
             'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30',
             'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P30_pow160',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30',
             'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P30_pow160', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R',
             'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30',
             'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P30_pow160',
             'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P30_pow160', 'R_pow16', 'R', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P30_pow160', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'TQ', 'V',
             'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P30_pow160', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P30_pow160', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
             'R', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30',
             'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P30_pow160', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30',
             'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P30_pow160', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
             'R', 'R', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P30_pow160', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P30_pow160', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30',
             'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P30_pow160', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P30_pow160', 'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
             'R', 'R', 'R', 'R', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P30_pow160', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16',
             'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30',
             'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P30_pow160', 'R_pow16', 'R_pow16', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30',
             'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P30_pow160',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
             'R', 'R', 'R', 'R', 'R', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'V', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V',
             'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'R_pow16', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'V', 'P30',
             'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P30', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10',
             'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P10', 'P30_pow160',
             'R_pow16', 'R_pow16', 'R_pow16', 'R_pow16', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
             'R', 'R', 'R', 'R', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ', 'TQ',
             'input'],
            ['output'],
            name='r15_einsum_1',
            domain='',
            equation='a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,oa,oa,...b,b,ob,...c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,oc,...d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,od,od,od,od,od,od,od,od,od,od,jb,jb,jb,jb,jb,jb,jb,jb,jc,jc,jc,jc,jc,jc,jc,jc,jd,jd,jd,jd,jd,jd,jd,jd,...f,jf,jf,jf,jf,jf,jf,jf,jf,of,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,...g,jg,jg,jg,jg,jg,jg,jg,jg,og,og,og,og,og,og,og,og,og,og,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,...h,jh,jh,jh,jh,jh,jh,jh,jh,oh,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,...i,ji,ji,ji,ji,ji,ji,ji,ji,oi,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,...k,jk,jk,jk,jk,jk,jk,jk,jk,ok,k,k,k,k,k,k,k,k,k,k,...l,jl,jl,jl,jl,jl,jl,jl,jl,ol,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,...m,jm,jm,jm,jm,jm,jm,jm,jm,om,om,om,om,om,om,om,om,om,om,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,...n,jn,jn,jn,jn,jn,jn,jn,jn,on,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,...p,jp,jp,jp,jp,jp,jp,jp,jp,op,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,...q,jq,jq,jq,jq,jq,jq,jq,jq,oq,q,q,q,q,q,q,q,q,q,q,q,q,...t,jt,jt,jt,jt,jt,jt,jt,jt,ot,ot,ot,ot,ot,ot,ot,ot,ot,ot,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,...A,jA,jA,jA,jA,jA,jA,jA,jA,oA,oA,oA,oA,oA,oA,oA,oA,oA,oA,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,...H,jH,jH,jH,jH,jH,jH,jH,jH,oH,oH,oH,oH,oH,oH,oH,oH,oH,oH,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,...O,jO,jO,jO,jO,jO,jO,jO,jO,oO,oO,oO,oO,oO,oO,oO,oO,oO,oO,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,...Q,jQ,jQ,jQ,jQ,jQ,jQ,jQ,jQ,oQ,oQ,oQ,oQ,oQ,oQ,oQ,oQ,oQ,oQ,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,...u,ju,ju,ju,ju,ju,ju,ju,ju,ou,ou,ou,ou,ou,ou,ou,ou,ou,ou,ou,ou,ou,ou,ou,ou,ou,ou,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,...e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,e,je,je,je,je,je,je,je,je,oe,oe,oe,oe,oe,oe,oe,oe,oe,oe,oe,oe,oe,oe,oe,oe,oe,oe,ru,re,...s,js,js,js,js,js,js,js,js,os,os,os,os,os,os,os,os,os,os,rs,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,...v,jv,jv,jv,jv,jv,jv,jv,jv,ov,ov,ov,ov,ov,ov,ov,ov,ov,ov,rv,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,...w,jw,jw,jw,jw,jw,jw,jw,jw,ow,ow,ow,ow,ow,ow,ow,ow,ow,ow,ow,ow,ow,ow,ow,ow,ow,ow,rw,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,...x,jx,jx,jx,jx,jx,jx,jx,jx,ox,ox,ox,ox,ox,ox,ox,ox,ox,ox,rx,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,...y,jy,jy,jy,jy,jy,jy,jy,jy,oy,oy,oy,oy,oy,oy,oy,oy,oy,oy,oy,oy,oy,oy,oy,oy,oy,oy,ry,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,...z,jz,jz,jz,jz,jz,jz,jz,jz,oz,oz,oz,oz,oz,oz,oz,oz,oz,oz,rz,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,...B,jB,jB,jB,jB,jB,jB,jB,jB,oB,oB,oB,oB,oB,oB,oB,oB,oB,oB,oB,oB,oB,oB,oB,oB,oB,oB,rB,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,...C,jC,jC,jC,jC,jC,jC,jC,jC,oC,oC,oC,oC,oC,oC,oC,oC,oC,oC,rC,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,...D,jD,jD,jD,jD,jD,jD,jD,jD,oD,oD,oD,oD,oD,oD,oD,oD,oD,oD,rD,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,...E,jE,jE,jE,jE,jE,jE,jE,jE,oE,oE,oE,oE,oE,oE,oE,oE,oE,oE,oE,oE,oE,oE,oE,oE,oE,oE,rE,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,...F,jF,jF,jF,jF,jF,jF,jF,jF,oF,oF,oF,oF,oF,oF,oF,oF,oF,oF,rF,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,...G,jG,jG,jG,jG,jG,jG,jG,jG,oG,oG,oG,oG,oG,oG,oG,oG,oG,oG,oG,oG,oG,oG,oG,oG,oG,oG,rG,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,...I,jI,jI,jI,jI,jI,jI,jI,jI,oI,oI,oI,oI,oI,oI,oI,oI,oI,oI,rI,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,...J,jJ,jJ,jJ,jJ,jJ,jJ,jJ,jJ,oJ,oJ,oJ,oJ,oJ,oJ,oJ,oJ,oJ,oJ,oJ,oJ,oJ,oJ,oJ,oJ,oJ,oJ,rJ,J,J,J,J,J,J,J,J,J,J,J,J,J,J,J,J,J,J,J,J,J,J,...K,jK,jK,jK,jK,jK,jK,jK,jK,oK,oK,oK,oK,oK,oK,oK,oK,oK,oK,rK,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,...L,jL,jL,jL,jL,jL,jL,jL,jL,oL,oL,oL,oL,oL,oL,oL,oL,oL,oL,rL,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,...M,jM,jM,jM,jM,jM,jM,jM,jM,oM,oM,oM,oM,oM,oM,oM,oM,oM,oM,rM,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,M,...N,jN,jN,jN,jN,jN,jN,jN,jN,oN,oN,oN,oN,oN,oN,oN,oN,oN,oN,oN,oN,oN,oN,oN,oN,oN,oN,rN,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,...P,jP,jP,jP,jP,jP,jP,jP,jP,oP,oP,oP,oP,oP,oP,oP,oP,oP,oP,oP,oP,oP,oP,oP,oP,oP,oP,rP,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,...R,jR,jR,jR,jR,jR,jR,jR,jR,oR,oR,oR,oR,oR,oR,oR,oR,oR,oR,rR,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,...S,jS,jS,jS,jS,jS,jS,jS,jS,oS,oS,oS,oS,oS,oS,oS,oS,oS,oS,oS,oS,oS,oS,oS,oS,oS,oS,rS,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,...T,jT,jT,jT,jT,jT,jT,jT,jT,oT,oT,oT,oT,oT,oT,oT,oT,oT,oT,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,...U,jU,jU,jU,jU,jU,jU,jU,jU,oU,oU,oU,oU,oU,oU,oU,oU,oU,oU,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,...V,jV,jV,jV,jV,jV,jV,jV,jV,oV,oV,oV,oV,oV,oV,oV,oV,oV,oV,oV,oV,oV,oV,oV,oV,oV,oV,rV,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,...Zrj->...orj',
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
            _tensor('P30', TensorProto.FLOAT, (30, 2), INIT_P30_0),
            _tensor('P10', TensorProto.FLOAT, (10, 2), INIT_P10_1),
            _tensor('R', TensorProto.FLOAT, (2,), INIT_R_2),
            _tensor('TQ', TensorProto.FLOAT, (2,), INIT_TQ_3),
            _tensor('R_pow16', TensorProto.FLOAT, (2,), INIT_R_POW16_4),
            _tensor('P30_pow160', TensorProto.FLOAT, (30, 2), INIT_P30_POW160_5),
        ],
        value_info=[
            _vi('V', TensorProto.FLOAT, [1, 2]),
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
