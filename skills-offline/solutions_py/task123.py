from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task123'
TASK_NUM = 123
KAGGLE = {'score': 20.24641, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 116
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task123_fused_colfact_m0_p120'
OPSETS = [('', 18)]


# P: FLOAT[2, 30], 60 value(s)
INIT_P_0 = [4.0, 3.8095240592956543, 3.6281182765960693, 3.455350875854492, 3.2908105850219727, 3.1341054439544678,
 2.9848625659942627, 2.842726230621338, 2.7073585987091064, 2.5784366130828857, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# H: FLOAT[2, 2, 2], 8 value(s)
INIT_H_1 = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

# C0: FLOAT[2, 2], 4 value(s)
INIT_C0_2 = [-4.698086261749268, 4.059011459350586, 5.883862495422363, -5.326345443725586]

# C1: FLOAT[2, 2], 4 value(s)
INIT_C1_3 = [4.698086261749268, -4.2569804191589355, 5.626141548156738, -5.883862495422363]

# C3: FLOAT[2, 2], 4 value(s)
INIT_C3_4 = [-3.055856943130493, 3.5788683891296387, 4.223257064819336, -4.861039161682129]

# C4: FLOAT[2, 2], 4 value(s)
INIT_C4_5 = [-4.8469557762146, 6.167316436767578, 5.8469557762146, -7.12190055847168]

# C5: FLOAT[2, 2], 4 value(s)
INIT_C5_6 = [3.4992828369140625, -4.698086261749268, 4.394079685211182, -5.883862495422363]

# C6: FLOAT[2, 2], 4 value(s)
INIT_C6_7 = [3.340893268585205, -4.698086261749268, 4.180269718170166, -5.883862495422363]

# C8: FLOAT[2, 2], 4 value(s)
INIT_C8_8 = [-6.633349418640137, 10.290426254272461, -9.42400074005127, 14.619807243347168]

# Q: FLOAT[2, 2, 2], 8 value(s)
INIT_Q_9 = [-0.976085364818573, 1.1449077129364014, 0.999894917011261, -1.1792482137680054, -1.4011210203170776, 1.0,
 1.3680092096328735, -1.0]

# W: FLOAT[10], 10 value(s)
INIT_W_10 = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# R2: FLOAT[2], 2 value(s)
INIT_R2_11 = [0.04570341855287552, -0.020716967061161995]


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
        # 0: Einsum inputs=100 outputs=1
        _node(
            'Einsum',
            ['H', 'P', 'P', 'H', 'H', 'H', 'P', 'P', 'Q', 'P', 'P', 'P', 'H', 'C8', 'P', 'H', 'P', 'P', 'H',
             'C6', 'Q', 'Q', 'P', 'H', 'P', 'P', 'H', 'C6', 'P', 'H', 'P', 'P', 'H', 'C5', 'P', 'H', 'P', 'P',
             'H', 'C4', 'P', 'H', 'P', 'P', 'H', 'C3', 'P', 'H', 'P', 'P', 'H', 'C0', 'C0', 'C0', 'C6', 'Q',
             'R2', 'P', 'H', 'P', 'P', 'H', 'C1', 'P', 'H', 'P', 'P', 'H', 'C0', 'P', 'H', 'P', 'P', 'input',
             'C4', 'H', 'P', 'H', 'P', 'H', 'Q', 'P', 'H', 'P', 'C0', 'C8', 'C8', 'C8', 'H', 'C6', 'C6', 'Q',
             'C0', 'C0', 'C3', 'C8', 'P', 'W', 'input', 'W'],
            ['output'],
            name='',
            domain='',
            equation='SqU,Ur,Wr,TqW,SsV,TsX,Vc,Xc,tST,Sj,Tj,Qr,PqQ,tP,Pj,PsR,Rc,Nr,MqN,tM,MMM,MMM,Mj,MsO,Oc,Kr,JqK,tJ,Jj,JsL,Lc,Hr,GqH,tG,Gj,GsI,Ic,Er,DqE,tD,Dj,DsF,Fc,Br,AqB,tA,Aj,AsC,Cc,xr,wqx,wt,wz,zw,tz,tzt,z,wj,wsy,yc,or,mqo,tm,mj,msp,pc,fr,eqf,te,ej,esg,gc,aj,nkjj,Za,aqb,br,asd,dc,qqs,quq,ur,uuv,vc,ti,ti,tt,tt,iiY,Yi,Yi,iiY,Yi,YY,ii,YY,ih,l,nlhh,k->nkrc',
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
            _tensor('P', TensorProto.FLOAT, (2, 30), INIT_P_0),
            _tensor('H', TensorProto.FLOAT, (2, 2, 2), INIT_H_1),
            _tensor('C0', TensorProto.FLOAT, (2, 2), INIT_C0_2),
            _tensor('C1', TensorProto.FLOAT, (2, 2), INIT_C1_3),
            _tensor('C3', TensorProto.FLOAT, (2, 2), INIT_C3_4),
            _tensor('C4', TensorProto.FLOAT, (2, 2), INIT_C4_5),
            _tensor('C5', TensorProto.FLOAT, (2, 2), INIT_C5_6),
            _tensor('C6', TensorProto.FLOAT, (2, 2), INIT_C6_7),
            _tensor('C8', TensorProto.FLOAT, (2, 2), INIT_C8_8),
            _tensor('Q', TensorProto.FLOAT, (2, 2, 2), INIT_Q_9),
            _tensor('W', TensorProto.FLOAT, (10,), INIT_W_10),
            _tensor('R2', TensorProto.FLOAT, (2,), INIT_R2_11),
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
