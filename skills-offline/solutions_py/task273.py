from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task273'
TASK_NUM = 273
KAGGLE = {'score': 20.522663, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 88
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task273_hosted_safe_champion_rebuild'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task273_hosted_safe_champion_rebuild'
OPSETS = [('', 18)]


# F: FLOAT[30, 2], 60 value(s)
INIT_F_0 = [0.876780092716217, 0.48089149594306946, 0.761214554309845, 0.6485001444816589, 0.6293655633926392, 0.7771093845367432,
 0.46716511249542236, 0.8841701149940491, 0.24469220638275146, 0.9696007966995239, -0.05009227246046066,
 0.9987446069717407, -0.28240591287612915, 0.9592950344085693, -0.46292832493782043, 0.8863956928253174,
 -0.623052716255188, 0.7821798324584961, -0.7634540796279907, 0.6458621025085449, -1.2579996585845947,
 1.7611994743347168, 1.4873979091644287, -1.863161563873291, -1.66279137134552, 1.8378220796585083, 1.784672737121582,
 -1.7095285654067993, -1.8383569717407227, 1.4900367259979248, 1.7692683935165405, -1.1733043193817139,
 1.7722976207733154, -0.9141324758529663, -1.810310959815979, 0.6669566631317139, -1.7300536632537842,
 0.3824329078197479, 1.8094148635864258, -0.13332530856132507, 1.7188538312911987, 0.12665238976478577,
 -1.796544075012207, -0.3971308171749115, -1.6918644905090332, -0.6233184933662415, 1.7842671871185303,
 0.9203062057495117, 1.512525200843811, 1.0030430555343628, -1.7669745683670044, -1.43217933177948, 1.7398653030395508,
 1.6666078567504883, -1.6323633193969727, -1.804190993309021, 1.4667655229568481, 1.8373167514801025,
 -1.2445762157440186, -1.7424067258834839]

# M: FLOAT[2, 2, 2], 8 value(s)
INIT_M_1 = [1.7009243965148926, -0.02190651185810566, -0.01813824661076069, 1.5001518726348877, -0.007176512852311134,
 -1.3520809412002563, 1.2809823751449585, 0.0064763459376990795]

# E: FLOAT[10, 2], 20 value(s)
INIT_E_2 = [-1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'M', 'M', 'F', 'F', 'F',
             'M', 'F', 'M', 'F', 'M', 'F', 'F', 'F', 'F', 'M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'F', 'F',
             'input', 'input', 'input', 'E', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'M', 'F',
             'F', 'F', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'E', 'input', 'E', 'E'],
            ['output'],
            name='',
            domain='',
            equation='nr,nr,nr,nr,nr,nr,nr,nr,nk,nk,nk,nk,nZ,nZ,ni,kFG,kvx,wF,wv,wH,rHI,wy,ryA,wD,iDE,pE,pI,pG,wB,iBC,pC,wt,itu,wf,ifs,qA,qu,qs,qx,...caq,...cap,...cbp,cz,aK,aM,aQ,aO,bY,bW,bU,bS,gRS,gTU,gLM,gJK,hJ,hR,hL,hT,jNO,hN,jVW,hV,mXY,hX,mPQ,hP,dg,dj,dj,dj,dj,dm,dm,dm,dm,dm,dm,dm,dm,dZ,dZ,lZ,...lhw,le,oe->...ohw',
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
            _tensor('F', TensorProto.FLOAT, (30, 2), INIT_F_0),
            _tensor('M', TensorProto.FLOAT, (2, 2, 2), INIT_M_1),
            _tensor('E', TensorProto.FLOAT, (10, 2), INIT_E_2),
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
