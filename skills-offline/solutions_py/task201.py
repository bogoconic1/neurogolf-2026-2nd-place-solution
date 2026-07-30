from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task201'
TASK_NUM = 201
KAGGLE = {'score': 19.506939, 'date': '2026-07-15'}
MEMORY_BYTES = 182
PARAMS = 61
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task201_cases30_selector_cost243'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# W: INT64[30], 30 value(s)
INIT_W_1 = [1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# CASES: STRING[30], 30 value(s)
INIT_CASES_2 = ['-9104701970819448833,-1417317917392897,-1,-1,9151314442808393727,-1,-1,-45195154071486465,-1,-1',
 '-8667256683421499393,-1,-45212882891177985,-1,8935141660694413311,-1,-1,-1,-1,-5651575867441153',
 '-9101823766593273857,-4225423185543169,-1,-1,9151313888766066687,-1,-1,-1,-45264694692282369,-1',
 '-9097893761962213377,-1,-3669083186790401,-1,9151314440652587007,-1,-49751595486806017,-1,-1,-1',
 '-8667256683421499393,-1,-5651575867441153,-1,8935141660694413311,-1,-1,-45212882891177985,-1,-1',
 '-8658223267686383617,-54255129079054337,-1,-1,8935141660694413311,-5642745414680577,-1,-1,-1,-1',
 '-8664934994549932033,-7903289580453889,-45282286878326785,-1,8935141093767380991,-1,-1,-1,-1,-1',
 '-9102911975653769217,-1,-1,-45300706114011137,9151314442808393727,-1,-3101761040547841,-1,-1,-1',
 '-8660422598829539329,-3408486046105601,-54289486133067777,-1,8935141093767380991,-1,-1,-1,-1,-1',
 '-9091078978225995265,-1,-1,-1423889250975745,9151314442816814847,-1,-1,-1,-58811575339843585,-1',
 '-9091719178808590337,-1,-1,-1,9151314442808393727,-1,-1,-1,-58740930361950209,-854333637787649',
 '-8658258452721696769,-1,-1,-1,8935141658488471551,-1,-54219942022610945,-1,-1,-5642745213353985',
 '-9102967140541005825,-1,-1,-1,9151314440652587007,-1,-1,-1,-3099579113275393,-45247720981528577',
 '-8667230774828728321,-1,-1,-1,8935141093767380991,-1,-1,-45247102506237953,-1,-5642693673746433',
 '-8669482128510550017,-1,-3390945869430785,-45248067800137729,8935141660694413311,-1,-1,-1,-1,-1',
 '-8651485768318976001,-1,-1,-1,8935141093767380991,-1,-3408486046105601,-1,-1,-63226316643631105',
 '-9085003146903552001,-1,-1,-1,9151314442808393727,-3101743860678657,-63209552044097537,-1,-1,-1',
 '-8667177998270595073,-1,-1,-1,8935141093767380991,-1,-45282286878326785,-1,-5660285859790849,-1',
 '-9097900709105369089,-1,-1,-1,9151313888766066687,-3662473232121857,-1,-1,-1,-49750702133608449',
 '-8651415229789372417,-1,-1,-3408537585713153,8935141658488471551,-1,-1,-63297372582576129,-1,-1',
 '-9085537561698238465,-2549806136295425,-63227074973794305,-1,9151314442808393727,-1,-1,-1,-1,-1',
 '-8658188185722421249,-1,-1,-1,8935141660694413311,-54290176549060609,-1,-5642779908636673,-1,-1',
 '-8667221602791849985,-1,-1,-1,8935141658488471551,-1,-1,-5651610025852929,-45247927139958785,-1',
 '-8667230466270035969,-1,-1,-5642745548898305,8935141660694413311,-1,-1,-45247930361184257,-1,-1',
 '-8658223129978994689,-1,-1,-5642745683116033,8935141660694413311,-1,-1,-1,-1,-54255266518007809',
 '-8655962741162377217,-1,-7903409839538177,-1,8935141658488471551,-1,-1,-54254988955746305,-1,-1',
 '-9107404742064996353,-1,-40796970389864449,-1,9151314442808393727,-1,-1,-1,-3112730353467393,-1',
 '-9090597583188918273,-58740996397072385,-1,-1,9151314440652587007,-1,-1975861049819137,-1,-1,-1',
 '-8669508620137267201,-1,-45212745989095425,-1,8935141660694413311,-1,-1,-3399776053755905,-1,-1',
 '-9081007812456218625,-1,-67765100643090433,-1,9151313888766066687,-1,-1,-2540971371790337,-1,-1']


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
            high=28.605682373046875,
            low=-26.08678436279297,
            seed=5402863.0,
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
            ['packed', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
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
            _tensor('W', TensorProto.INT64, (30,), INIT_W_1),
            _tensor('CASES', TensorProto.STRING, (30,), INIT_CASES_2),
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
