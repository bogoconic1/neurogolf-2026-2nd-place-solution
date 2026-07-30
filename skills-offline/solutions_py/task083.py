from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task083'
TASK_NUM = 83
KAGGLE = {'score': 20.872866, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 62
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task083_updated_21_fast_mem0_params62_fast22'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task083_updated_21_fast'
OPSETS = [('', 13)]


# P: FLOAT[2, 30], 60 value(s)
INIT_P_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.125, -0.25, -1.0, 0.5, -0.5, -0.125, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# T: FLOAT[2], 2 value(s)
INIT_T_1 = [1.0, -2.0]


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
        # 0: Einsum inputs=167 outputs=1
        _node(
            'Einsum',
            ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T',
             'T', 'T', 'T', 'T', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'T', 'P', 'P',
             'P', 'T', 'P', 'T', 'T', 'T', 'P', 'T', 'T', 'P', 'T', 'T', 'T', 'P', 'P', 'P', 'T', 'T', 'T', 'P',
             'P', 'P', 'T', 'T', 'P', 'T', 'P', 'P', 'T', 'P', 'P', 'P', 'P', 'T', 'P', 'T', 'T', 'P', 'P', 'T',
             'T', 'T', 'T', 'P', 'P', 'P', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T',
             'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'input', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'T', 'T', 'T', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'T', 'P', 'P', 'P', 'T', 'T', 'T',
             'T', 'P', 'P', 'P', 'P', 'P', 'P', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'P', 'P', 'T', 'T', 'T', 'T',
             'T', 'T', 'P', 'P', 'P', 'T', 'T', 'P'],
            ['output'],
            name='signed_power_cover_62p_fast_22factor',
            domain='',
            equation='c,c,c,c,c,c,c,c,g,g,g,g,g,g,g,a,a,a,a,a,d,d,d,d,dh,ah,ch,gh,ch,gh,ah,dr,ar,gr,cr,er,eh,e,kh,kr,kh,k,kh,k,k,k,kr,i,i,ih,i,i,i,ir,lr,lh,l,l,l,lh,fr,fr,f,f,fh,f,br,bh,j,jr,jr,jr,jh,j,jr,m,m,mr,mr,m,m,m,m,mh,mh,mh,q,q,q,q,q,q,q,q,q,q,q,q,p,p,p,p,p,p,p,n,n,n,n,n,n,...hw,nw,qw,qw,qw,qw,pw,uw,uw,uw,uw,u,u,u,qs,ns,us,ps,ps,ow,os,x,xs,xs,xw,z,z,z,z,zs,zs,zw,vw,tw,yw,t,t,t,t,t,t,t,ts,ts,t,y,y,y,y,y,ys,ys,vs,v,v,vs->...rs',
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
            _tensor('T', TensorProto.FLOAT, (2,), INIT_T_1),
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
