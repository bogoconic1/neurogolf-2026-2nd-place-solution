from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task238'
TASK_NUM = 238
KAGGLE = {'score': 19.511062, 'date': '2026-07-14'}
MEMORY_BYTES = 182
PARAMS = 60
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task238_tied_coordinate'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# CASES: STRING[29], 29 value(s)
INIT_CASES_1 = ['-37400979448321,-137977921537,-30820685316097,-7169,-8690728961,-1,-1,-1,-68719738881,-1',
 '9053307994791706111,-8944148859957805057,-1,-36240592602136577,-1,-569572893786113,-1,-268467201,-8796231434241,-1',
 '-18590646556967425,-1,-1063937,-1,-1,-16906090788683777,-2224826744833,-1,-35184506306561,-70782138974209',
 '-18599339032592897,-70782675845121,-2224826744833,-16897294695661569,-15361,-1,-1,-1,-35287587880961,-1',
 '9055560036328373759,-36240731114831873,-569607253524481,-1,-1080321,-1,-1,-1,-8865349369857,-8946400659771490305',
 '-37469363380737,-8623620097,-7169,-1,-30820685316097,-1,-1,-1,-134479873,-138246356993',
 '-18630160121348609,-1,-1,-70782138974209,-2224826744833,-1,-1,-1588225,-4467034423297,-16897294695661569',
 '-37435136287233,-30786325577729,-1,-1,-1,-138246356993,-1,-531457,-68720787457,-8623620097',
 '-18590629176558081,-2207713984513,-16906090788683777,-1,-15361,-70782138974209,-1,-1,-35219000524801,-1',
 '9072466040081317375,-8944166452143849473,-36240730041090049,-1,-1,-1,-569590006546433,-1,-19149095050674177,-1080321',
 '-18590663600783873,-2207646875649,-539649,-70782138974209,-1,-1,-1,-1,-35184642883585,-16906090788683777',
 '9079234599099006463,-565157667405825,-1,-1,-272137217,-36240730041090049,-8950904259398860801,-1,-19184278881435649,-1',
 '-18625848039981569,-70782138974209,-1,-1,-16906090788683777,-2207713984513,-539649,-1,-136577025,-1',
 '-37469429703169,-8690728961,-1,-1,-7169,-30820685316097,-1,-138246356993,-1048577,-1',
 '9074634483506839039,-271088641,-565192027144193,-8946418251957534721,-36170498735865857,-1,-1,-1,-19140367677128705,-1',
 '9049939159952621055,-1,-1,-36240592602136577,-1,-8939662852516478977,-1,-569572826677249,-1125969167646721,-1604609',
 '9071348661861449215,-8944148859957805057,-569590073655297,-271612929,-1,-1,-36240455163183105,-1,-18049583557115905,-1',
 '-37469563920897,-7169,-30820685316097,-8690728961,-1,-1,-137977921537,-1,-135266305,-1',
 '-37400979448321,-7169,-1,-30820685316097,-8690728961,-1,-1,-1,-68719738881,-137977921537',
 '-37469698138625,-137977921537,-1,-1,-30820685316097,-1,-8690728961,-1,-1048577,-7169',
 '-18586247973585409,-1063937,-1,-1,-2224826744833,-70782675845121,-1,-16906090788683777,-39582552817665,-1',
 '-18581970522227201,-1,-70644700020737,-539649,-16914886881705985,-1,-2207713984513,-1,-35219000262657,-1',
 '-18608169887744513,-2224826744833,-1,-16888498602639361,-1,-70782138974209,-1,-539649,-35253362098177,-1',
 '9054442759375257087,-1,-36240592602136577,-569572893786113,-1,-271088641,-8944166452143849473,-1,-1125968626319361,-1',
 '-18634627019981313,-1,-1063937,-2224826744833,-1,-16897294695661569,-70782138974209,-1,-136314881,-1',
 '-18595027153338881,-16897294695661569,-1,-1,-2207646875649,-1,-70782138974209,-539649,-39617183350785,-1',
 '-37469496549889,-8623620097,-7169,-1,-1,-138246356993,-30820685316097,-1,-1310721,-1',
 '-37400576532993,-1,-30820685316097,-531457,-138246356993,-1,-8690728961,-1,-68853694465,-1',
 '9076982902567173631,-565191960035329,-8948652459585175553,-1,-556033,-1,-36240730041090049,-1,-19184348142239745,-1']

# P: INT64[30], 30 value(s)
INIT_P_2 = [64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


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
            ['rng_f'],
            name='',
            domain='',
            dtype=10,
            high=13.812538146972656,
            low=-16.865692138671875,
            seed=559349184.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['rng_f'],
            ['case_i'],
            name='',
            domain='',
            to=6,
        ),
        # 2: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['CASES', 'case_i'],
            ['case_string'],
            name='',
            domain='',
            axis=0,
        ),
        # 3: StringSplit inputs=1 outputs=2
        _node(
            'StringSplit',
            ['case_string'],
            ['parts', 'lengths'],
            name='',
            domain='',
            delimiter=',',
            maxsplit=9,
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
        # 5: Einsum inputs=10 outputs=1
        _node(
            'Einsum',
            ['packed', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['output'],
            name='',
            domain='',
            equation='bc,r,r,r,r,r,r,r,r,w->bcrw',
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
            _tensor('CASES', TensorProto.STRING, (29,), INIT_CASES_1),
            _tensor('P', TensorProto.INT64, (30,), INIT_P_2),
        ],
        value_info=[
            _vi('rng_f', TensorProto.FLOAT16, [1]),
            _vi('case_i', TensorProto.INT32, [1]),
            _vi('case_string', TensorProto.STRING, [1]),
            _vi('parts', TensorProto.STRING, [1, 10]),
            _vi('lengths', TensorProto.INT64, [1]),
            _vi('packed', TensorProto.INT64, [1, 10]),
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
