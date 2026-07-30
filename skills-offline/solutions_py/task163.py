from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task163'
TASK_NUM = 163
KAGGLE = {'score': 19.82385, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 177
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task163_repeated_power_homotopy_factorization'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task163_exact_gauge_tied_diag_p177'
OPSETS = [('', 22)]


# yellow: FLOAT[10], 10 value(s)
INIT_YELLOW_0 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ch: FLOAT[10, 2], 20 value(s)
INIT_CH_1 = [-1.2593191862106323, -1.0044760704040527, -0.8410642147064209, -3.046971559524536, -0.8128409385681152,
 -3.2535054683685303, -0.7267851233482361, -3.825477361679077, -0.9302005171775818, -2.651564359664917,
 -1.6786662340164185, -3.463371992111206, -0.8679603934288025, -2.9017672538757324, -0.8214259147644043,
 -3.1360740661621094, -0.7922645807266235, -3.354429244995117, -0.7695602178573608, -3.546978712081909]

# F: FLOAT[30, 3], 90 value(s)
INIT_F_2 = [0.0, 1.0, 2.0, 0.5, 1.0, 1.5, 1.0, 1.0, 1.0, 0.0, 1.0, -1.0, -0.5, 1.0, 1.5, 0.0, 1.0, 1.0, 0.5, 1.0, 0.5, 0.0, 1.0,
 -1.0, -1.0, 1.0, 1.0, -0.5, 1.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# LC0: FLOAT[3, 3], 9 value(s)
INIT_LC0_3 = [-1.0, -1.0, 1.0, 0.5, 1.0, -0.5, -1.0, -2.0, 1.0]

# LC1: FLOAT[3, 3], 9 value(s)
INIT_LC1_4 = [1.0, 1.7763568394002505e-15, -1.0, 1.0, 1.0, -1.0, 1.0, 1.7763568394002505e-15, -1.0]

# Dmap: FLOAT[3, 3, 3], 27 value(s)
INIT_DMAP_5 = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -3.0, -1.0, -2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, -1.0, -0.0, -0.0,
 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

# L1: FLOAT[3, 2, 2], 12 value(s)
INIT_L1_6 = [7.778833389282227, -4.86553430557251, -0.7739312052726746, 0.4613560438156128, 3.1915574073791504, -1.276443362236023,
 0.6298362016677856, -0.2530966103076935, 2.1112051010131836, -4.161974906921387, 9.436137199401855,
 -0.13701540231704712]


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
        # 0: Einsum inputs=84 outputs=1
        _node(
            'Einsum',
            ['LC1', 'LC0', 'LC0', 'LC0', 'Dmap', 'F', 'F', 'F', 'LC0', 'F', 'LC1', 'input', 'yellow', 'ch', 'F',
             'LC0', 'LC0', 'Dmap', 'F', 'LC1', 'LC0', 'F', 'F', 'LC0', 'LC1', 'LC1', 'LC0', 'LC0', 'LC0',
             'Dmap', 'F', 'F', 'F', 'F', 'LC0', 'F', 'LC1', 'input', 'ch', 'ch', 'F', 'LC0', 'LC0', 'Dmap',
             'LC1', 'LC0', 'F', 'F', 'F', 'F', 'LC0', 'LC1', 'L1', 'L1', 'Dmap', 'Dmap', 'LC1', 'LC0', 'LC0',
             'LC0', 'Dmap', 'F', 'F', 'F', 'LC0', 'LC1', 'F', 'F', 'LC1', 'LC1', 'LC1', 'LC0', 'LC0', 'LC0',
             'Dmap', 'F', 'F', 'F', 'F', 'LC0', 'LC1', 'F', 'ch', 'ch'],
            ['output'],
            name='task163_exact_gauge_tied_diag_p177',
            domain='',
            equation='sB,BB,sA,AA,sCC,xB,xA,xP,iP,xQ,iQ,...axy,a,aq,yD,tD,DD,tCC,yE,tE,EE,yU,yV,jU,jV,sH,HH,sG,GG,sCC,uH,uG,uI,uW,rW,uX,rX,...cuv,cp,cz,vJ,tJ,JJ,tCC,tK,KK,vK,vL,vY,vZ,lY,lZ,kpq,kzn,kid,kje,dN,NN,dM,MM,dCC,hN,hM,hb,rb,rf,hf,hO,Ok,Tk,eS,SS,eR,RR,eCC,wS,wR,wT,wg,lg,lm,wm,oq,on->...ohw',
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
            _tensor('yellow', TensorProto.FLOAT, (10,), INIT_YELLOW_0),
            _tensor('ch', TensorProto.FLOAT, (10, 2), INIT_CH_1),
            _tensor('F', TensorProto.FLOAT, (30, 3), INIT_F_2),
            _tensor('LC0', TensorProto.FLOAT, (3, 3), INIT_LC0_3),
            _tensor('LC1', TensorProto.FLOAT, (3, 3), INIT_LC1_4),
            _tensor('Dmap', TensorProto.FLOAT, (3, 3, 3), INIT_DMAP_5),
            _tensor('L1', TensorProto.FLOAT, (3, 2, 2), INIT_L1_6),
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
