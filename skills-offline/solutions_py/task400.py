from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task400'
TASK_NUM = 400
KAGGLE = {'score': 19.768891, 'date': '2026-07-15'}
MEMORY_BYTES = 86
PARAMS = 101
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# CASES: STRING[30], 30 value(s)
INIT_CASES_1 = ['-22348899,-254262063923201,-481036337153,-458152214529', '-23331333,6413688544451100671,-51338280961,-274877906945',
 '-4163035241869485447,-341114929713,-1,-274877906945', '-39846417,-7493844369531731969,-1,-274877906945',
 '-26818565,-274877906945,-219714420737,-549728995323', '-540757537171,-549689804321,-71269613569,-293668388865',
 '-32474707,-274877906945,-34634157,-274877906945', '-274886330447,-6629589472281362433,-1012625,-274877906945',
 '-5779085,-274877906945,-47378857985,-549750034803', '-46145537,-274877906945,-5762355173464735745,-274945015809',
 '-535659064865,-549690429853,-2162755,-274877906945', '-25462873,6999181235020890111,-1,-274877906945',
 '-17240207,-274877906945,6468915089368940543,-1729963287065395201', '-17453699,-274877906945,-1,4797636249550585855',
 '-1048621,-274877906945,-66060243,-274877906945', '-1082233,297481941865725951,-1,-274877906945',
 '-274900594253,-549722259457,-10867123,-274877906945', '-112219717921,-549742116575,-1,-274877906945',
 '293033042940771327,-311385128961,-63,-274877906945', '-37776577,-274877906945,-1,-8062797656440700929',
 '-141885452355,-549738500029,-1,-274877906945', '-4143981809518604361,-467950108673,4314013036416335871,-496605593601',
 '-12346965874593929,-416947372033,-16580609,-274877906945', '-46345235,-274877906945,-379701952513,-549709468653',
 '-39914721,-7475068559220015105,-549755813889,-549688705025', '-62530067,-1258610135681466369,-1,-274877906945',
 '-16024137,-274877906945,-1,4404681239143907327', '-6303979,-274877906945,4165403044806131711,-2432578714668630017',
 '-47124577,-274877906945,-394935664641,-297519600090770335',
 '-17762891,5476376872004616191,-461308428289,-275079233537']

# V: INT64[30], 30 value(s)
INIT_V_2 = [16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# A: INT64[4, 10], 40 value(s)
INIT_A_3 = [274877906944, 8192, 1, 0, 0, 0, 0, 0, 0, 0, 0, 67108864, 0, 274877906944, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 274877906944, 1, 0, 0, 8192, 0, 0, 0, 0, 0, 0, 0, 274877906944, 1, 67108864]


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
        # 0: RandomUniformLike inputs=1 outputs=1
        _node(
            'RandomUniformLike',
            ['LIKE'],
            ['kf'],
            name='',
            domain='',
            dtype=10,
            high=28.605682373046875,
            low=-26.08678436279297,
            seed=5402863.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['kf'],
            ['ki'],
            name='',
            domain='',
            to=6,
        ),
        # 2: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['CASES', 'ki'],
            ['cs'],
            name='',
            domain='',
            axis=0,
        ),
        # 3: StringSplit inputs=1 outputs=2
        _node(
            'StringSplit',
            ['cs'],
            ['parts', 'lens'],
            name='',
            domain='',
            delimiter=',',
            maxsplit=3,
        ),
        # 4: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['parts'],
            ['packed'],
            name='',
            domain='',
            to=7,
        ),
        # 5: Einsum inputs=8 outputs=1
        _node(
            'Einsum',
            ['packed', 'A', 'V', 'V', 'V', 'V', 'V', 'V'],
            ['output'],
            name='',
            domain='',
            equation='bl,lc,r,r,r,r,r,w->bcrw',
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.INT64, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('LIKE', TensorProto.FLOAT16, (1,), INIT_LIKE_0),
            _tensor('CASES', TensorProto.STRING, (30,), INIT_CASES_1),
            _tensor('V', TensorProto.INT64, (30,), INIT_V_2),
            _tensor('A', TensorProto.INT64, (4, 10), INIT_A_3),
        ],
        value_info=[
            _vi('kf', TensorProto.FLOAT16, [1]),
            _vi('ki', TensorProto.INT32, [1]),
            _vi('cs', TensorProto.STRING, [1]),
            _vi('parts', TensorProto.STRING, [1, 4]),
            _vi('lens', TensorProto.INT64, [1]),
            _vi('packed', TensorProto.INT64, [1, 4]),
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
