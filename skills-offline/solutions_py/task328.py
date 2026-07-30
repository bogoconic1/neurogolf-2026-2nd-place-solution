from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task328'
TASK_NUM = 328
KAGGLE = {'score': 19.763558, 'date': '2026-07-14'}
MEMORY_BYTES = 20
PARAMS = 168
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 9
PRODUCER_NAME = 'NeuroGolf-expand-simplify-recompress'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task328_m36_p238'
OPSETS = [('', 17)]


# CHS: FLOAT[2, 10], 20 value(s)
INIT_CHS_0 = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# SMODE: FLOAT[2, 2, 2], 8 value(s)
INIT_SMODE_1 = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

# PAIR: FLOAT[2, 2, 2], 8 value(s)
INIT_PAIR_2 = [1.0, 0.0, -0.5799999833106995, -0.0, 1.0, -1.0, -0.41999998688697815, -0.0]

# REL: FLOAT[2, 2, 2], 8 value(s)
INIT_REL_3 = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]

# V: FLOAT[2, 2, 30], 120 value(s)
INIT_V_4 = [0.5293639302253723, 0.06352367997169495, 0.007622841279953718, 0.0009147409582510591, 0.0001097688582376577,
 1.3172262697480619e-05, 1.5806714372956776e-06, 1.8968057702295482e-07, 2.27616681058862e-08, 2.731400217115265e-09,
 3.277680071800404e-10, 3.933215905749243e-11, 4.7198590522046224e-12, 5.663830949381721e-13, 6.796596434526653e-14,
 8.15591592471989e-15, 9.787118422015066e-16, 1.1744541947599402e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 4.4113664627075195, 36.761390686035156, 306.3449401855469, 2552.87451171875, 21273.943359375, 177282.875,
 1477357.25, 12311311.0, 102594264.0, 854952192.0, 7124601856.0, 59371683840.0, 494764032000.0, 4123033600000.0,
 34358614294528.0, 286321813749760.0, 2386019957604352.0, 1.988349955722445e+16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# DB: FLOAT[2, 2], 4 value(s)
INIT_DB_5 = [1.8890595436096191, 0.22668713331222534, 1.0, -1.0]


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
            ['Nf'],
            name='grid_size',
            domain='',
            axes=[1, 2, 3],
            keepdims=0,
        ),
        # 1: Pow inputs=2 outputs=1
        _node(
            'Pow',
            ['DB', 'Nf'],
            ['DP'],
            name='powered_2x2',
            domain='',
        ),
        # 2: Einsum inputs=111 outputs=1
        _node(
            'Einsum',
            ['PAIR', 'PAIR', 'DB', 'DB', 'V', 'V', 'input', 'PAIR', 'PAIR', 'DB', 'DB', 'V', 'V', 'CHS', 'CHS',
             'PAIR', 'PAIR', 'DB', 'DB', 'V', 'V', 'input', 'PAIR', 'PAIR', 'DB', 'DB', 'V', 'V', 'REL', 'PAIR',
             'PAIR', 'DB', 'DB', 'V', 'V', 'input', 'PAIR', 'PAIR', 'DB', 'DB', 'V', 'V', 'CHS', 'REL', 'PAIR',
             'PAIR', 'SMODE', 'REL', 'SMODE', 'SMODE', 'PAIR', 'REL', 'REL', 'SMODE', 'PAIR', 'DP', 'SMODE',
             'REL', 'SMODE', 'SMODE', 'SMODE', 'V', 'DB', 'REL', 'REL', 'DP', 'REL', 'V', 'SMODE', 'DP',
             'SMODE', 'REL', 'V', 'REL', 'REL', 'REL', 'DP', 'SMODE', 'SMODE', 'SMODE', 'DB', 'V', 'DP', 'V',
             'SMODE', 'REL', 'REL', 'REL', 'REL', 'DP', 'V', 'SMODE', 'SMODE', 'SMODE', 'DB', 'DP', 'V',
             'SMODE', 'REL', 'SMODE', 'REL', 'REL', 'REL', 'DP', 'SMODE', 'SMODE', 'SMODE', 'DB', 'V', 'input',
             'PAIR'],
            ['output'],
            name='four_term_classifier',
            domain='',
            equation='mKP,mKm,PP,PP,PPE,PPE,neDE,jKU,jKj,UU,UU,UUD,UUD,Ke,Kd,iKS,iKi,SS,SS,SSB,SSB,ndAB,gKu,gKg,uu,uu,uuA,uuA,qli,lKN,lKl,NN,NN,NNC,NNC,noRC,kKO,kKk,OO,OO,OOR,OOR,bo,bMy,bKy,bKy,bMq,qkg,bMq,bMq,bMv,vgj,vim,tKw,KKK,Kg,wKt,tqa,tFK,aFK,gFK,KgH,FF,agZ,ZaZ,tZ,Lag,taH,stL,Ki,swQ,Qci,KiW,wqc,ciV,VcV,wV,iGK,wGK,cGK,GG,wcW,Kj,KjH,sXz,zfj,Xyf,fjp,pfp,Xp,XfH,fIK,jIK,XIK,II,Km,KmW,XKY,Yyh,sYr,rhm,hmT,ThT,YT,mJK,YJK,hJK,JJ,YhW,nxHW,sss->noHW',
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
            _tensor('CHS', TensorProto.FLOAT, (2, 10), INIT_CHS_0),
            _tensor('SMODE', TensorProto.FLOAT, (2, 2, 2), INIT_SMODE_1),
            _tensor('PAIR', TensorProto.FLOAT, (2, 2, 2), INIT_PAIR_2),
            _tensor('REL', TensorProto.FLOAT, (2, 2, 2), INIT_REL_3),
            _tensor('V', TensorProto.FLOAT, (2, 2, 30), INIT_V_4),
            _tensor('DB', TensorProto.FLOAT, (2, 2), INIT_DB_5),
        ],
        value_info=[
            _vi('Nf', TensorProto.FLOAT, [1]),
            _vi('DP', TensorProto.FLOAT, [2, 2]),
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
