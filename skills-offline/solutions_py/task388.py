from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task388'
TASK_NUM = 388
KAGGLE = {'score': 19.634024, 'date': '2026-07-14'}
MEMORY_BYTES = 46
PARAMS = 168
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task388_t3factor'
OPSETS = [('', 17)]


# V3: FLOAT[2, 30], 60 value(s)
INIT_V3_0 = [1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# V4: FLOAT[2, 30], 60 value(s)
INIT_V4_1 = [0.0, 1.0, 2.0, 2.0, 0.0, -4.0, -8.0, -8.0, 0.0, 16.0, 32.0, 32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, -2.0, -4.0, -4.0, 0.0, 8.0, 16.0, 16.0, 0.0, -32.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# G3: FLOAT[2, 2, 2], 8 value(s)
INIT_G3_2 = [-1.0, -1.0, 0.0, -1.0, -1.0, -0.0, -1.0, -1.0]

# E: FLOAT[3, 10], 30 value(s)
INIT_E_3 = [1.270240306854248, -2.102881908416748, -2.1805124282836914, -0.2943524122238159, 1.709270715713501,
 -1.0146222114562988, -0.3376203179359436, -1.3963901996612549, 1.754908800125122, -1.7740287780761719,
 1.5709123611450195, 1.5842914581298828, 1.7843513488769531, 1.6668071746826172, 2.3248212337493896, 1.0586735010147095,
 1.3932820558547974, 1.5435805320739746, 0.7435309886932373, 1.4672693014144897, -1.5933825969696045,
 -0.01883600279688835, 1.0599385499954224, -2.2131731510162354, 2.864110231399536, 1.0775022506713867,
 1.869106411933899, -1.5919214487075806, -0.10200092196464539, -0.9665843844413757]

# U: FLOAT[2, 3], 6 value(s)
INIT_U_4 = [-0.7753150463104248, 1.1601471900939941, -0.3809080421924591, -0.06844694167375565, -0.3099583089351654,
 0.6129366755485535]

# H4: FLOAT[2, 2], 4 value(s)
INIT_H4_5 = [-1.0, -1.0, -1.0, 1.0]


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
        # 0: ReduceL2 inputs=1 outputs=1
        _node(
            'ReduceL2',
            ['input'],
            ['nf'],
            name='',
            domain='',
            axes=[1, 2, 3],
            keepdims=1,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['nf'],
            ['n8'],
            name='',
            domain='',
            to=2,
        ),
        # 2: Div inputs=2 outputs=1
        _node(
            'Div',
            ['n8', 'n8'],
            ['one8'],
            name='',
            domain='',
        ),
        # 3: ConvInteger inputs=2 outputs=1
        _node(
            'ConvInteger',
            ['one8', 'n8'],
            ['shift'],
            name='',
            domain='',
            kernel_shape=[1, 1],
            pads=[0, 1, 0, 0],
            strides=[1, 1],
        ),
        # 4: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['V3', 'shift'],
            ['P3'],
            name='',
            domain='',
            axis=1,
        ),
        # 5: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['V4', 'shift'],
            ['P4'],
            name='',
            domain='',
            axis=1,
        ),
        # 6: Einsum inputs=94 outputs=1
        _node(
            'Einsum',
            ['P3', 'G3', 'U', 'U', 'E', 'input', 'E', 'input', 'V3', 'V3', 'G3', 'H4', 'G3', 'P3', 'V3', 'V3',
             'P4', 'H4', 'H4', 'V4', 'H4', 'H4', 'V4', 'V4', 'V4', 'P4', 'H4', 'H4', 'H4', 'H4', 'V4', 'H4',
             'H4', 'H4', 'H4', 'P4', 'H4', 'H4', 'H4', 'H4', 'H4', 'H4', 'H4', 'H4', 'H4', 'H4', 'H4', 'H4',
             'V4', 'G3', 'V3', 'G3', 'P3', 'V3', 'G3', 'H4', 'G3', 'P3', 'V3', 'V3', 'P4', 'H4', 'H4', 'V4',
             'H4', 'H4', 'V4', 'V4', 'V4', 'H4', 'H4', 'H4', 'H4', 'H4', 'H4', 'H4', 'H4', 'H4', 'H4', 'H4',
             'P4', 'H4', 'V4', 'P4', 'H4', 'H4', 'H4', 'H4', 'V4', 'H4', 'H4', 'H4', 'H4', 'E'],
            ['output'],
            name='',
            domain='',
            equation='dxxxp,def,xl,xy,yb,...bsc,la,...arc,fr,jr,iZj,ii,ghi,gxxxp,eR,hR,kxxxp,km,kn,nr,mn,mm,mR,wr,Er,txxxp,tu,tv,uv,uu,uR,Zv,wv,Zw,ww,zxxxp,ZE,EE,ED,ZD,ZD,DD,DB,ZB,zB,AB,zA,AA,AR,ZZZ,Hc,FGH,Fxxxq,Lc,KZL,KK,IJK,Ixxxq,GC,JC,Mxxxq,MN,MO,Oc,NO,NN,NC,Tc,Yc,ZY,YY,UY,UU,VY,VV,ZY,ZZ,ZX,XX,UV,Uxxxq,VV,VC,Pxxxq,PQ,PS,QS,QQ,QC,ZS,TS,ZT,TT,lo->...oRC',
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
            _tensor('V3', TensorProto.FLOAT, (2, 30), INIT_V3_0),
            _tensor('V4', TensorProto.FLOAT, (2, 30), INIT_V4_1),
            _tensor('G3', TensorProto.FLOAT, (2, 2, 2), INIT_G3_2),
            _tensor('E', TensorProto.FLOAT, (3, 10), INIT_E_3),
            _tensor('U', TensorProto.FLOAT, (2, 3), INIT_U_4),
            _tensor('H4', TensorProto.FLOAT, (2, 2), INIT_H4_5),
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
