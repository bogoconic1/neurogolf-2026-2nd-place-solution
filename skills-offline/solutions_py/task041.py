from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task041'
TASK_NUM = 41
KAGGLE = {'score': 20.617973, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task041_folded_core_p81_emptyS'
OPSETS = [('', 18)]


# F: FLOAT[2, 30], 60 value(s)
INIT_F_0 = [0.20142599940299988, 0.20142599940299988, 0.20142599940299988, 0.20142599940299988, 0.20142599940299988,
 0.20142599940299988, 0.20142599940299988, 0.20142599940299988, 0.20142599940299988, 0.20142599940299988,
 -0.3357169032096863, -0.29833683371543884, 0.17710541188716888, -0.23713645339012146, -0.287662148475647,
 -0.28556951880455017, -0.2997731566429138, -0.3044002652168274, -0.30280834436416626, -0.2876117527484894,
 0.30360323190689087, -0.2876340448856354, 0.2997940182685852, -0.026159241795539856, 0.2965256869792938,
 0.3576624095439911, 0.34821394085884094, -0.23441830277442932, 0.23621417582035065, 0.19085891544818878,
 -0.027192508801817894, -0.021149728447198868, -0.015106949023902416, -0.009064169600605965, -0.0030213899444788694,
 0.0030213899444788694, 0.009064169600605965, 0.015106949023902416, 0.021149728447198868, 0.027192508801817894,
 -0.266731858253479, 0.1201493889093399, 0.336351215839386, 0.3036039471626282, -0.20849299430847168,
 -0.20371855795383453, 0.09281821548938751, 0.08799617737531662, 0.07486449182033539, -0.21321475505828857,
 -0.26297369599342346, -0.20127882063388824, 0.28360214829444885, -0.36632320284843445, -0.258599191904068,
 0.12908430397510529, 0.18022261559963226, 0.31155717372894287, 0.2459046095609665, -0.1877617985010147]

# E: FLOAT[2, 10], 20 value(s)
INIT_E_1 = [65330940477440.0, 4.700381284977869e+16, 3.728061248202342e+16, 8.27736920602706e+16, 1.8451916520003994e+17,
 1.88990793090859e+17, 2.183559139903406e+17, 3.319890835113247e+17, 7.42106302196482e+17, 1.6591372077907313e+18,
 -6402432049348608.0, -2.2091790321409065e+18, -8.20546217636266e+17, -8.082767673819136e+17, -6.946779747791667e+17,
 -1.5587061664422298e+17, 1.3447914547943834e+17, 4.38815502962262e+17, 1.2375889851868774e+18, 3.0480982952357396e+18]


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
        # 0: Einsum inputs=93 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'input', 'input', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'input', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'input', 'F', 'F', 'F', 'E', 'E', 'F', 'F', 'F', 'E', 'E', 'F', 'F', 'F'],
            ['output'],
            name='',
            domain='',
            equation='eM,eM,eM,eM,dM,dM,pM,pN,bN,bN,bN,bN,aN,aN,ax,dx,fx,fO,fO,gO,gO,gO,gO,uO,by,ey,nchx,nchy,iy,iP,iP,uP,jP,jP,jP,jP,pt,ut,ut,BR,BR,BR,BR,AR,AR,QR,Ar,nchr,Cr,Er,CS,CS,QS,DS,DS,DS,DS,Ds,Bs,nchs,Gs,ET,ET,FT,FT,FT,FT,VT,GU,GU,VU,HU,HU,HU,HU,gw,jw,Fw,Hw,nkhw,Qm,Vm,Vm,Kc,Jc,Jz,Jz,Iz,Io,Lo,Kl,Ll,Ll->nohw',
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
            _tensor('F', TensorProto.FLOAT, (2, 30), INIT_F_0),
            _tensor('E', TensorProto.FLOAT, (2, 10), INIT_E_1),
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
