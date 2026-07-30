from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task046'
TASK_NUM = 46
KAGGLE = {'score': 19.506939, 'date': '2026-07-14'}
MEMORY_BYTES = 182
PARAMS = 61
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task046_selector30_exact'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# CASES: STRING[30], 30 value(s)
INIT_CASES_1 = ['2270906044488089599,-1,-1080863910568919041,-15728641,-1153449270188179457,-1,-36029896580923393,-1,-1,-1',
 '2305808907206918143,-1,-1153449270188179457,-1,-1080863910568919041,-1,-1,-1,-54043745301037057,-1',
 '139332862187929599,-1,-281487861612545,-72059793262510081,-1,-1,-63050807100047361,-1,-1,-4026531841',
 '562932774076415,-1,-66228395704321,-1,-1,-1,-1,-492581209243649,-1,-4123168604161',
 '8727923276120063,-1006632961,-1,-1,-1,-1,-281491082838017,-1,-1,-1924178903041',
 '1152904445013524479,-527765581332481,-864708720641179649,-1,-1,-1,-1,-1,-270215977642229761,-1',
 '35976007341047807,-1,-492583356727297,-1,-1,-1,-7696581394433,-1,-8797972070401,-1',
 '1143897520653467647,-252201579132747777,-1,-1,-1,-1,-18014948290461697,-1,-865236486222512129,-1',
 '143838111145656319,-281487861612545,-1,-72059793262510081,-1,-1,-1,-1,-4026531841,-1',
 '35896846933032959,-3758096385,-422216760033281,-7696984047617,-1,-1,-1,-1,-1,-1',
 '4503548290138111,-527765581332481,-1,-30786459795457,-1,-1,-3377905878958081,-4123235713025,-1,-1',
 '72055378103238655,-1,-1,-65970898993153,-1,-1,-1,-1,-492581209243649,-1',
 '4239716572463103,-1,-1,-1,-8797972070401,-1,-8658654068737,-1,-281490009096193,-1',
 '7061591426139684863,-4612108230892453889,-1,-1,-1,-1,-2305913381179097089,-1,-13194944839681,-1',
 '2305815504276684799,-1,-108088590080147457,-1008810714577502209,-1,-1,-1153449270188179457,-1,-1,-1',
 '35888057416679423,-1,-281487861612545,-16492674416641,-123147449794561,-1,-1,-1,-1,-1',
 '1125848371429375,-1,-1,-1,-34634616274945,-1,-527765581332481,-1,-1,-481039482881',
 '4179305259224530943,-1,-1,-1,-1,-1,-4035326423341203457,-1,-1,-422216760033281',
 '62804103650082815,-1,-1,-18015223143202817,-1,-1,-26392037163009,-36317981461970945,-1,-1',
 '8935033891450781695,-6917951240106147841,-1945585825349632001,-1,-1,-1,-1,-1,-1,-1',
 '8214495334533955583,-3458799698192629761,-4612139017218031617,-1,-1,-1,-1,-1,-1,-1',
 '450096079694725119,-281491082838017,-1,-1,-288239173050040321,-1,-1,-144123434413064193,-1,-1',
 '8682818019673505791,-1729399849633185793,-1,-1,-6917951240106147841,-1,-503316481,-1,-1,-1',
 '18001188213489663,-17592991350785,-1,-531614073356289,-1,-1,-1,-1,-1,-1',
 '35967211415797759,-492584430469121,-4398986035201,-1,-3298568437761,-1,-1,-1,-1,-1',
 '-108360186497859585,-1,-1,2305561534236983295,-1,-1,-1,-1,-2199123918849,-2161732219184349185',
 '8142420150189031423,-3458799699266371585,-4612108230892453889,-8796898328577,-1,-1,-1,-1,-6597136875521,-1',
 '9007182888566783,-4123168604161,-492581209243649,-1,-1,-1,-52777095004161,-1,-13194407968769,-1',
 '1152658171051900927,-1,-281491082838017,-1,-864708721178050561,-1,-1,-1,-270215977642229761,-1',
 '2251789078364159,-8246337208321,-1,-422216760033281,-1,-1,-481036337153,-131943542816769,-1,-1']

# V: INT64[30], 30 value(s)
INIT_V_2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


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
        # 5: Einsum inputs=17 outputs=1
        _node(
            'Einsum',
            ['packed', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V'],
            ['output'],
            name='',
            domain='',
            equation='bc,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,w->bcrw',
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
