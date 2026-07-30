from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task156'
TASK_NUM = 156
KAGGLE = {'score': 20.072746, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 138
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task156_direct_einsum_138'
OPSETS = [('', 18)]


# E: FLOAT[30, 3], 90 value(s)
INIT_E_0 = [0.9795371294021606, 0.9771468639373779, -0.21256528794765472, 0.9795371294021606, 0.9369497299194336,
 0.34946417808532715, 0.9795371294021606, 0.599277675151825, 0.8005412220954895, 0.9795371294021606, 0.0713391825556755,
 0.9974521398544312, 0.9795371294021606, -0.4792490005493164, 0.8776789903640747, 0.9795371294021606,
 -0.8776789903640747, 0.4792490005493164, 0.9795371294021606, -0.9974521398544312, -0.0713391825556755,
 0.9795371294021606, -0.8005412220954895, -0.599277675151825, 0.9795371294021606, -0.34946417808532715,
 -0.9369497299194336, 0.9795371294021606, 0.21256528794765472, -0.9771468639373779, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0]

# A: FLOAT[3, 2], 6 value(s)
INIT_A_1 = [-0.4329526424407959, 0.0, 0.0, 1.0, 0.0, 1.0]

# V: FLOAT[3, 3, 2], 18 value(s)
INIT_V_2 = [-0.549754798412323, 0.33438652753829956, -0.0007989277364686131, -0.00042857739026658237, 0.021250994876027107,
 -0.011571252718567848, 0.7191160917282104, 0.010150227695703506, 6.611074059037492e-05, -6.342944743664702e-06,
 -0.026822609826922417, -0.0003713212499860674, -0.5914422273635864, -0.3751189112663269, -0.0006988828536123037,
 0.0002974501985590905, 0.022670432925224304, 0.013215212151408195]

# B: FLOAT[10, 2], 20 value(s)
INIT_B_3 = [-0.00017213411047123373, -33.7542839050293, -0.00014282524352893233, -37.55352783203125, 0.9976161122322083,
 -0.2864558696746826, 0.0, 0.0, 0.947174608707428, -0.43761172890663147, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0]

# M: FLOAT[2, 2], 4 value(s)
INIT_M_4 = [-7.344100475311279, -28.417545318603516, -7.531891822814941, 0.00020540821424219757]


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
        # 0: Einsum inputs=74 outputs=1
        _node(
            'Einsum',
            ['A', 'A', 'V', 'B', 'B', 'M', 'B', 'input', 'E', 'A', 'A', 'A', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
             'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'A', 'A', 'A', 'A', 'B',
             'input', 'B', 'B', 'M', 'V', 'V', 'A', 'A', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
             'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'A', 'E', 'A', 'A', 'V', 'B', 'B', 'M', 'B'],
            ['output'],
            name='',
            domain='',
            equation='yC,yC,yKE,dC,dE,DE,dD,nduv,ux,xq,bB,fq,hf,hb,hi,rb,ri,hg,rg,hm,rm,hz,rz,rt,ht,hj,rj,he,re,hl,rl,ha,ra,aA,pA,pF,pF,cF,ncrs,cG,cH,GH,pLH,kpq,pN,MN,sM,sQ,wM,wQ,wV,sV,sR,wR,sX,wX,sU,wU,sS,wS,sW,wW,sO,sT,wT,OP,wO,kI,kI,kZY,oI,oY,JY,oJ->nohw',
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
            _tensor('E', TensorProto.FLOAT, (30, 3), INIT_E_0),
            _tensor('A', TensorProto.FLOAT, (3, 2), INIT_A_1),
            _tensor('V', TensorProto.FLOAT, (3, 3, 2), INIT_V_2),
            _tensor('B', TensorProto.FLOAT, (10, 2), INIT_B_3),
            _tensor('M', TensorProto.FLOAT, (2, 2), INIT_M_4),
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
