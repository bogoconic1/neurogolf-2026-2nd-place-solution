from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task122'
TASK_NUM = 122
KAGGLE = {'score': 20.569183, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 84
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task122_r28_branch_projective_k4'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task122_r28_branch_projective_k4'
OPSETS = [('', 13)]


# Q: FLOAT[2, 30], 60 value(s)
INIT_Q_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -20.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.550000011920929, 1.100000023841858, 1.6500000953674316, 2.200000047683716,
 2.75, 3.3000001907348633, 3.8500001430511475, 4.400000095367432, 4.950000286102295, 5.5, 6.050000190734863,
 6.600000381469727, 7.150000095367432, 7.700000286102295, 8.25, 8.800000190734863, 9.350000381469727, 9.90000057220459,
 10.449999809265137, 11.0, -51.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# OutBasis: FLOAT[2, 10], 20 value(s)
INIT_OUTBASIS_1 = [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000031710769e-30, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0]

# K: FLOAT[2, 2], 4 value(s)
INIT_K_2 = [1.0, -1.0, 1.0, 0.0]


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
        # 0: Einsum inputs=86 outputs=1
        _node(
            'Einsum',
            ['K', 'K', 'K', 'K', 'K', 'K', 'K', 'Q', 'Q', 'input', 'OutBasis', 'input', 'Q', 'OutBasis',
             'OutBasis', 'K', 'OutBasis', 'K', 'K', 'Q', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'Q', 'input',
             'input', 'Q', 'K', 'OutBasis', 'K', 'OutBasis', 'OutBasis', 'Q', 'input', 'input', 'Q', 'K', 'K',
             'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K',
             'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'Q', 'Q', 'OutBasis',
             'OutBasis', 'input', 'Q', 'Q', 'OutBasis'],
            ['output'],
            name='',
            domain='',
            equation='is,iR,Rs,sz,sj,Rj,ij,ze,je,...dec,gd,...abc,ib,ga,gx,JB,gy,PB,PB,zv,JP,AJ,AC,CJ,fJ,Af,Cf,zf,Am,...kmr,...kur,Du,NN,Nk,Nn,nk,nX,GV,...kUV,...kUY,qY,PE,PE,LE,LP,DL,zl,lL,Dl,Fl,FL,DF,PQ,QP,TP,TP,TQ,TQ,QO,QO,IO,IG,zH,HI,HG,HK,IK,KG,QZ,QZ,MZ,zS,SM,Sq,St,Mq,Mt,tq,Ch,Fh,Tp,Tp,...phw,Kw,tw,To->...ohw',
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
            _tensor('Q', TensorProto.FLOAT, (2, 30), INIT_Q_0),
            _tensor('OutBasis', TensorProto.FLOAT, (2, 10), INIT_OUTBASIS_1),
            _tensor('K', TensorProto.FLOAT, (2, 2), INIT_K_2),
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
