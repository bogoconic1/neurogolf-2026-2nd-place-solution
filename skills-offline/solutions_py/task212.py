from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task212'
TASK_NUM = 212
KAGGLE = {'score': 20.212508, 'date': '2026-07-14'}
MEMORY_BYTES = 0
PARAMS = 120
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task212_updated_v5'
PRODUCER_VERSION = '5'
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task212_updated_v5'
OPSETS = [('', 18)]


# E: FLOAT[2, 2, 10], 40 value(s)
INIT_E_0 = [-0.17138312757015228, -0.067294642329216, 0.40370041131973267, 0.24245847761631012, -0.3789302408695221,
 0.016986163333058357, 0.014270301908254623, 0.23137745261192322, -0.24784429371356964, 0.023373452946543694,
 -0.16299423575401306, -0.06692219525575638, -0.13043369352817535, 0.06357116252183914, 0.25625574588775635,
 0.7373173236846924, -0.12990833818912506, -0.7710860371589661, 0.4377889037132263, 0.020853139460086823,
 -0.22916142642498016, 0.5312433838844299, 0.0032473208848387003, -0.522713840007782, 0.1331111639738083,
 -0.07302393764257431, -0.15614689886569977, 0.06777765601873398, 0.14112471044063568, -0.06393210589885712,
 0.5635379552841187, -0.39591822028160095, -0.27515852451324463, 0.18400444090366364, 0.23767125606536865,
 0.3541468679904938, -0.01172318123281002, -0.33496734499931335, -0.3581516742706299, 0.049109507352113724]

# P: FLOAT[2, 30], 60 value(s)
INIT_P_1 = [1.441640019416809, 1.441640019416809, 1.441640019416809, 1.441640019416809, 1.441640019416809, 1.441640019416809,
 1.441640019416809, 1.441640019416809, 1.441640019416809, 1.441640019416809, -1.3924123048782349, -1.3924123048782349,
 -1.3924123048782349, -1.3924123048782349, -1.3924123048782349, -1.3924123048782349, -1.3924123048782349,
 -1.3924123048782349, -1.3924123048782349, -1.3924123048782349, 0.1507735401391983, 0.1507721245288849,
 0.1507721245288849, 0.1507721245288849, 0.1507721245288849, 0.1507721245288849, 0.1507721245288849, 0.1507721245288849,
 0.1507721245288849, 0.1507721245288849, 0.0, 0.6447209715843201, 1.2894419431686401, 1.934162974357605,
 2.5788838863372803, 3.223604917526245, 3.86832594871521, 4.513047218322754, 5.1577677726745605, 5.802488803863525,
 -8.039413452148438, -1.0512052774429321, -1.0512142181396484, -1.0512142181396484, -1.051211953163147,
 -1.0512124300003052, -1.051213026046753, -1.0512096881866455, -1.0512126684188843, -1.0512118339538574,
 -1.0512094497680664, -1.0512107610702515, -1.0512104034423828, -1.051213264465332, -1.0512123107910156,
 -1.0512135028839111, -1.051214337348938, -1.051214575767517, -1.0512142181396484, -1.0512140989303589]

# D: FLOAT[2, 2, 2], 8 value(s)
INIT_D_2 = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0]

# C: FLOAT[3, 2, 2], 12 value(s)
INIT_C_3 = [-1.5679423809051514, 2.3494949340820312, 3.960800886154175, 2.354043483734131, 2.163736343383789, 5.894242286682129,
 -3.0257954597473145, 2.295980453491211, 2.063990354537964, -6.145397663116455, 4.098991870880127, -0.6197364330291748]


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
        # 0: Einsum inputs=55 outputs=1
        _node(
            'Einsum',
            ['P', 'P', 'P', 'P', 'D', 'D', 'P', 'P', 'P', 'D', 'D', 'P', 'P', 'D', 'D', 'P', 'P', 'P', 'P', 'P',
             'D', 'P', 'D', 'P', 'P', 'D', 'P', 'P', 'input', 'E', 'P', 'input', 'E', 'C', 'C', 'C', 'D', 'P',
             'P', 'P', 'P', 'P', 'D', 'D', 'P', 'C', 'C', 'C', 'E', 'E', 'E', 'input', 'E', 'E', 'input'],
            ['output'],
            name='task212_v5_balanced_Escratch_U_synthesis',
            domain='',
            equation='xW,yZ,yZ,yZ,xGH,xIJ,Hh,Jh,Lh,yKL,yMN,Nh,Ph,zOP,zQR,Rh,Is,Ms,Qs,Us,wUV,Vh,wST,Th,Cs,lCD,Dh,Fh,...fhc,euf,Bs,...qsc,iXq,dlt,dlt,diX,tEF,Gr,Kr,Or,Sr,Er,klt,kAB,Ar,plt,plt,pmY,mYa,mYa,jva,...grc,jvg,jvo,...obc->...orc',
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
            _tensor('E', TensorProto.FLOAT, (2, 2, 10), INIT_E_0),
            _tensor('P', TensorProto.FLOAT, (2, 30), INIT_P_1),
            _tensor('D', TensorProto.FLOAT, (2, 2, 2), INIT_D_2),
            _tensor('C', TensorProto.FLOAT, (3, 2, 2), INIT_C_3),
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
