from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task331'
TASK_NUM = 331
KAGGLE = {'score': 20.24641, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 116
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task331_p116_absorbed_helper'
OPSETS = [('', 12)]


# F: FLOAT[30, 3], 90 value(s)
INIT_F_0 = [0.9795371294021606, 1.0, 0.0, 0.9795371294021606, 0.8412535190582275, 0.5406408309936523, 0.9795371294021606,
 0.4154150187969208, 0.9096319675445557, 0.9795371294021606, -0.1423148363828659, 0.9898214340209961,
 0.9795371294021606, -0.6548607349395752, 0.7557495832443237, 0.9795371294021606, -0.9594929814338684,
 0.28173255920410156, 0.9795371294021606, -0.9594929814338684, -0.28173255920410156, 0.9795371294021606,
 -0.6548607349395752, -0.7557495832443237, 0.9795371294021606, -0.1423148363828659, -0.9898214340209961,
 0.9795371294021606, 0.4154150187969208, -0.9096319675445557, -0.5036842823028564, 0.5189602375030518,
 2.6600682735443115, -1.8796765804290771, 1.5643121004104614, 1.8412350416183472, 2.3920764923095703,
 0.7507501244544983, -1.2212384939193726, -2.444094181060791, -1.316644310951233, 1.1941252946853638,
 -1.2310787439346313, 0.3062886595726013, -0.799265444278717, -1.2531747817993164, -0.8004697561264038,
 -0.7424092292785645, -1.35677170753479, 1.1529899835586548, -0.8471663594245911, 2.7677161693573, -1.9756487607955933,
 -2.192025661468506, 0.5400922298431396, -0.5697245001792908, -2.6700713634490967, -2.206733226776123,
 0.2123613953590393, 1.4370194673538208, -1.219001054763794, 1.0086781978607178, -0.8678019642829895,
 -1.4839963912963867, -0.718241810798645, 1.0617040395736694, -1.7271372079849243, 1.283644676208496,
 1.4281083345413208, 1.6358760595321655, -1.2080458402633667, 0.8846018314361572, -1.2188289165496826,
 -1.3664370775222778, 0.8196197748184204, -3.0232086181640625, 2.3957176208496094, 2.3541994094848633,
 2.2134170532226562, 1.447741150856018, -1.160761833190918, -1.251516580581665, 0.05695817992091179, -0.774556577205658,
 -1.1704241037368774, 1.4398515224456787, 0.23625119030475616, 2.7731001377105713, -2.3417882919311523,
 -2.182276725769043]

# B: FLOAT[10, 2], 20 value(s)
INIT_B_1 = [0.00561846187338233, 0.2206897735595703, 4.898653507232666, -2.322831392288208, 0.07967124879360199, 1.097483515739441,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07967124879360199, -0.6293987035751343, 0.07967124879360199, 1.6565725803375244,
 0.07967124879360199, 1.9778647422790527, 0.0, 0.0]

# A: FLOAT[3, 2], 6 value(s)
INIT_A_2 = [-27.161115646362305, 0.0317842923104763, 11.310226440429688, -0.07522694021463394, -7.180160045623779,
 0.2874712646007538]


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
        # 0: Einsum inputs=58 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input', 'B', 'B', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input', 'B', 'A', 'B', 'B', 'A'],
            ['output'],
            name='output',
            domain='',
            equation='jZ,jZ,jZ,jZ,jG,jG,jJ,xJ,xq,xq,iq,iz,iz,iz,iz,ig,ig,vY,vB,vB,TK,TE,sG,sE,sB,wE,wB,wZ,sA,wA,VD,wD,sD,ncrs,cC,cH,rg,rd,re,rb,ra,Ud,hz,ha,hd,he,Pe,Pk,hb,fb,fb,fX,nmhw,ol,ql,ou,ou,qu->nohw',
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
            _tensor('F', TensorProto.FLOAT, (30, 3), INIT_F_0),
            _tensor('B', TensorProto.FLOAT, (10, 2), INIT_B_1),
            _tensor('A', TensorProto.FLOAT, (3, 2), INIT_A_2),
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
