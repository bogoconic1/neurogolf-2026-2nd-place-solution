from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task190'
TASK_NUM = 190
KAGGLE = {'score': 20.263802, 'date': '2026-07-10'}
MEMORY_BYTES = 0
PARAMS = 114
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task190_updated_v21_builder'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task190_updated_v21'
OPSETS = [('', 12)]


# P: FLOAT[3, 30], 90 value(s)
INIT_P_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.079803228378296, -1.0780125856399536, -1.0763366222381592,
 -1.1747223138809204, 0.5037585496902466, -1.078452229499817, -0.8236534595489502, -0.7788914442062378,
 -0.9922655820846558, -0.3829881548881531, 0.48591142892837524, 0.3638877272605896, -0.7938624620437622,
 -1.0794150829315186, 0.31649866700172424, 0.3193735182285309, 0.3298105001449585, -1.0800062417984009,
 0.3177710473537445, -0.859038233757019, 1.0, 0.9009688496589661, 0.6234897971153259, 0.22252093255519867,
 -0.22252093255519867, -0.6234897971153259, -0.9009688496589661, -1.0, -0.9009688496589661, -0.6234897971153259,
 0.7266165614128113, 0.7265651226043701, 0.7267550826072693, -1.0856868028640747, -0.22937683761119843,
 0.7427246570587158, 0.2294083833694458, -0.3108350932598114, -0.39477798342704773, -1.2327907085418701,
 1.0246646404266357, 0.29860925674438477, 0.9357756972312927, 0.7265128493309021, -0.1811315268278122,
 -0.1834971308708191, -0.17401251196861267, 0.7263695001602173, -0.17873035371303558, -0.05898985639214516, 0.0,
 0.4338837265968323, 0.7818315029144287, 0.9749279022216797, 0.9749279022216797, 0.7818315029144287, 0.4338837265968323,
 1.2246468525851679e-16, -0.4338837265968323, -0.7818315029144287, -0.034641414880752563, -0.04072118550539017,
 -0.048018816858530045, 0.0508749783039093, -1.771705985069275, -0.03528987243771553, 1.5872995853424072,
 1.8024017810821533, -1.4647271633148193, 1.1294710636138916, 0.8320455551147461, -1.5317305326461792,
 -0.3013826310634613, -0.036577221006155014, -0.616295337677002, -0.6177834868431091, -0.609366238117218,
 -0.03360018506646156, -0.6166942119598389, -0.5217872262001038]

# Rel: FLOAT[2, 3], 6 value(s)
INIT_REL_1 = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]

# route: FLOAT[10], 10 value(s)
INIT_ROUTE_2 = [-9.999999717180685e-10, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# C: FLOAT[2, 2, 2], 8 value(s)
INIT_C_3 = [0.0695897787809372, 1.0, 0.09968232363462448, 0.07180730253458023, -0.5696814060211182, 1.0, 0.4854653477668762,
 -0.02627399004995823]


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
            ['P', 'P', 'P', 'Rel', 'P', 'P', 'Rel', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'Rel', 'P', 'P', 'Rel', 'C', 'Rel', 'Rel', 'P', 'P', 'P', 'P', 'P', 'P', 'Rel', 'Rel', 'P', 'P',
             'P', 'Rel', 'P', 'P', 'Rel', 'P', 'P', 'P', 'Rel', 'Rel', 'P', 'P', 'input', 'route', 'Rel', 'P',
             'P', 'Rel', 'Rel', 'P', 'P', 'input', 'route', 'P', 'P', 'Rel', 'P', 'P', 'P', 'P', 'input',
             'input', 'route'],
            ['output'],
            name='task190_v21_direct_cubic_signed_logits',
            domain='',
            equation='Lz,Lz,Lz,dL,Mz,Mz,eM,sz,sz,sT,sQ,tQ,tQ,tO,tq,tq,Rq,Rq,Rq,aR,Sq,Sq,bS,EFG,Et,GH,tI,HI,HI,HI,HI,HI,Ff,GJ,fK,JK,JK,gN,NP,fP,hW,fo,Wo,Wo,gm,ai,iX,mX,pCXY,C,hn,nY,jY,bj,dk,mU,kU,pDUV,D,nV,lV,el,ir,kr,jw,lw,pZrw,pcAB,c->pcrw',
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
            _tensor('P', TensorProto.FLOAT, (3, 30), INIT_P_0),
            _tensor('Rel', TensorProto.FLOAT, (2, 3), INIT_REL_1),
            _tensor('route', TensorProto.FLOAT, (10,), INIT_ROUTE_2),
            _tensor('C', TensorProto.FLOAT, (2, 2, 2), INIT_C_3),
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
