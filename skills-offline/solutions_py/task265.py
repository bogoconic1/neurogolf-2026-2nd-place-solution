from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task265'
TASK_NUM = 265
KAGGLE = {'score': 20.29952, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 110
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'expand_simplify_recompress_channel_rowsum'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task265_m110'
OPSETS = [('', 12)]


# F: FLOAT[30, 3], 90 value(s)
INIT_F_0 = [1.0, 1.0, 0.0, 1.0, 0.945817232131958, 0.3246994614601135, 1.0, 0.789140522480011, 0.614212691783905, 1.0,
 0.5469481348991394, 0.8371664881706238, 1.0, 0.24548548460006714, 0.9694002866744995, 1.0, -0.0825793445110321,
 0.9965844750404358, 1.0, -0.4016954302787781, 0.915773332118988, 1.0, -0.6772815585136414, 0.7357239127159119, 1.0,
 -0.8794737458229065, 0.47594738006591797, 1.0, -0.9863613247871399, 0.1645945906639099, 1.0, -0.9863613247871399,
 -0.1645945906639099, 1.0, -0.8794737458229065, -0.47594738006591797, 1.0, -0.6772815585136414, -0.7357239127159119,
 1.0, -0.4016954302787781, -0.915773332118988, 1.0, -0.0825793445110321, -0.9965844750404358, 1.0, 0.24548548460006714,
 -0.9694002866744995, 1.0, 0.5469481348991394, -0.8371664881706238, 1.0, 0.789140522480011, -0.614212691783905, 1.0,
 0.945817232131958, -0.3246994614601135, -2.3947203159332275, -0.056585345417261124, 0.3033685088157654,
 4.740096569061279, 0.10550816357135773, -0.1206187829375267, -2.31290864944458, -0.18430735170841217,
 -0.3357785642147064, -2.3016626834869385, 0.899224579334259, -0.8846233487129211, -2.3719301223754883,
 -0.6443508863449097, 0.9380238652229309, -2.173382520675659, -0.9200418591499329, -0.756601870059967,
 -2.319208860397339, -0.8240331411361694, 0.4465223550796509, -2.3120996952056885, -0.19492706656455994,
 -0.3194662630558014, -2.482961416244507, 1.2333426475524902, 1.1270577907562256, -2.2610983848571777,
 -0.1029529795050621, 0.323493629693985, -2.3122177124023438, 0.21729755401611328, -0.5456033945083618]

# A: FLOAT[2, 10], 20 value(s)
INIT_A_1 = [1.0, 0.0, 13.870165824890137, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.1799999475479126, 0.0, -16.36679458618164, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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
        # 0: Einsum inputs=91 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'A', 'A', 'A',
             'A', 'A', 'input', 'input', 'A', 'A', 'input', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'A', 'A', 'input', 'A', 'A', 'A', 'A'],
            ['output'],
            name='output',
            domain='',
            equation='Ya,Yb,Yg,ha,hg,hb,rb,hj,ri,hi,rg,ra,rj,hl,rl,hm,rm,hn,rn,hx,rx,hy,ry,hz,rz,hA,rA,hB,rB,hC,rC,hD,rD,hE,rE,hF,rF,tq,tf,tf,td,td,...frs,...drw,te,te,...ehs,sN,wN,sR,wR,wJ,sJ,sM,wM,wQ,sQ,wP,sP,sK,wK,wL,sL,wU,sU,wV,sV,wT,sT,wW,sW,sO,wO,sS,wS,sH,sI,sG,wH,wG,ZG,ZH,ZI,wI,tc,tc,...chw,to,to,pc,po->...ohw',
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
            _tensor('A', TensorProto.FLOAT, (2, 10), INIT_A_1),
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
