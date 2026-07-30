from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task069'
TASK_NUM = 69
KAGGLE = {'score': 18.862273, 'date': '2026-07-14'}
MEMORY_BYTES = 374
PARAMS = 89
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task069_p89'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# CASES: STRING[28], 28 value(s)
INIT_CASES_1 = ['-576160585628760577,-1,-1,-1,-299067162772481,-1,-1099511889921,-1,-1,-1;-5530408024444747777,-1,-1,-1,-3620903119837200385,-1,9151311144281964543,-1,-1,-1',
 '-576460752214206977,-1,-83886081,-5329921,-1,-1,-1,-1,-1,-1;316659650805759,-1,-33554433,-316659617234945,-1,-1,-1,-1,-1,-1',
 '-576448095571672577,-1,-1,-1,-11007464308737,-1649267441665,-1,-1,-1,-1;316659650805759,-1,-1,-1,-316659617234945,-33554433,-1,-1,-1,-1',
 '-264586112761833985,-5243905,-1,-1,-1,-288230380446679041,-1,-4503668347113473,-1,-19140590742552577;1531236998730186751,-72058487393746945,-1,-1,-1,-1048577,-1,-1441154217221029889,-1,-18024294114344961',
 '-575897217394798081,-563534370967553,-1,-1,-1,-1,-1,-537657345,-1,-1;16383,-1,-1,-1,-1,-1,-1,-1,-1,-1',
 '-576157287091221505,-1,-1,-1,-1,-1,-303465209301505,-2899969,-1,-1;1441151882134306815,-1,-1,-1,-1,-1,-1375731713,-1441151880758558721,-1,-1',
 '-81627742784790529,-20266198611170305,-329325722751485953,-1,-135528961,-1,-145241088020447233,-1,-1,-1;-8457021219093331969,-576636683290673153,9078693898825498623,-1,-9007199288295425,-1,-36028797153181697,-1,-1,-1',
 '-576460752166974977,-1,-1,-1,-134217729,-1,-1,-2230273,-1,-1;-9074673942513303553,-1,-1,-1,-8800387989505,-1,-1,9074682742901309439,-1,-1',
 '-576178727282614273,-1,-1,-282024732956161,-287852545,-1,-1,-1,-1,-1;-8466757945017745409,-1,-1,9043228043170021375,-576470098152259585,-1,-1,-1,-1,-1',
 '-576460734049811969,-1,-1,-1,-1,-1,-17179869185,-1073741825,-1,-1;1173328443791163391,-1,-1,-1,-2251800887951361,-1,-1152921506754330625,-18014398643699713,-1,-140737505165313',
 '-576460717364829697,-1,-1,-570425345,-1,-1,-34368167937,-1,-1,-1;612489549893894143,-1,-1,-36028797018963969,-1,-1,-576460752874913793,-1,-1,-1',
 '-575908247709950977,-1,-1,-163277476724737,-1,-147457,-1,-1,-1,-389227116599809;1495337239855103,-1,-1,-1196268651020289,-1,-67108865,-1,-1,-1,-299068521709569',
 '-430083868663337473,-1,-1,-1,-1,-1,-146375784126357505,-1099513728001,-1,-1;-9079247915238408193,-1,-1,-1,-1,-1,9223363240753233919,-144115325514809345,-1,-1',
 '-499756622055077377,-1,-1,-1,-1,-72198331530479617,-1,-4505798717865985,-1,-1;9147945467265023,-1,-1,-1,-1,-9007199388958721,-1,-140746078289921,-1,-1',
 '-499897359610672129,-1,-1,-1,-72059793065378305,-1,-1,-4503599627372545,-1,-1;72057679937290239,-1,-1,-1,-72057611217797121,-1,-1,-68719476737,-1,-1',
 '-211668632726369281,-1,-72058143797968897,-1,-1,-292733975779084801,-1,-1,-1,-1;1459166280341798911,-1,-288230377225453569,-1,-1,-1170935903116328961,-1,-1,-1,-1',
 '-576161684855176705,-1,-16385,-1,-1,-299067179794433,-1,-268435457,-1,-1;-7908310894340063233,-1,-18023194602504193,-1,-1,7926335188454342655,-1,-1099511758849,-1,-1',
 '-576174879263145473,-1,-278529,-1,-1,-1,-1,-1,-1,-285873039998977;118800033012269055,-1,-654311425,-1,-1,-1,-1,-1,-1,-118800032357941249',
 '-576460473130507777,-1,-1,-1,-1,-4294975489,-274877939713,-1,-1,-1;4573968371564543,-1,-1,-1,-1,-70368744177665,-4503599627370497,-1,-1,-1',
 '-576460404411072001,-1,-1,-73014444033,-1,-1,-274877906945,-1,-1,-1;-8502794713358237697,-1,-1,9079256573763387391,-1,-1,-576461860405133313,-1,-1,-1',
 '-528610006259990017,-1,-1,-45035996276326401,-2251799813685249,-1,-1,-1,-1,-562949953421313;-8646895135468978177,-1,-1,8646904481322090495,-549760008193,-1,-1,-1,-1,-8796093087745',
 '-267401209257806849,-1,-1,-373249,-20266199415783425,-1,-288793343629459457,-1,-1,-1;9483288494079,-1,-1,-9483287789569,-32769,-1,-655361,-1,-1,-1',
 '-575822962510752769,-1,-1,-1,-1,-73014477313,-1,-1,-1,-637716778192897;-8646910725668716545,-1,-1,-1,-1,-9126805505,-1,-1,-1,8646910734795538431',
 '-215994076757915137,-1,-1,-360325903680571393,-140771864936449,-1,-1,-1,-1,-1;1531224165632196607,-1,-1,-1459166571325816833,-72057594306363393,-1,-1,-1,-1,-1',
 '-576460752258823681,-1,-1,-1,-1,-8388609,-1,-2656257,-1,-33554433;-4665729212143878145,-1,-1,-1,-1,-2594073385365405697,-1,8556839290192003071,-1,-1297036692682702849',
 '-575908139799161857,-442106753728513,-1,-110505750532609,-1,-1,-1,-1,-1,-1;3028811498303537151,-578712561243914241,-1,-2450098937059606529,-1,-1,-1,-1,-1,-1',
 '-576321298966205441,-24537183855105,-43997644982273,-1,-549755813889,-70368752566273,-1,-1,-1,-1;3038381647522840575,-721138901204860929,-2392537841139713,-1,-9007199263129601,-2305843009213693953,-1,-1,-1,-1',
 '-576281907180862977,-175921860444161,-1,-1,-2923262115841,-1,-1,-1,-1,-1;-342261883916500993,-5764610271813304321,-1,-1,6106872155729821695,-1,-1,-1,-1,-1']

# R: INT64[2, 30], 60 value(s)
INIT_R_2 = [2, 128, 512, 8, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 256, 16,
 1, 64, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


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
            high=13.686503410339355,
            low=-18.553632736206055,
            seed=7649488.0,
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
            ['case_strings'],
            name='',
            domain='',
            axis=0,
        ),
        # 3: StringSplit inputs=1 outputs=2
        _node(
            'StringSplit',
            ['case_strings'],
            ['pair_parts', 'pair_lengths'],
            name='',
            domain='',
            delimiter=';',
            maxsplit=1,
        ),
        # 4: StringSplit inputs=1 outputs=2
        _node(
            'StringSplit',
            ['pair_parts'],
            ['parts', 'lengths'],
            name='',
            domain='',
            delimiter=',',
            maxsplit=9,
        ),
        # 5: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['parts'],
            ['packed'],
            name='',
            domain='',
            to=7,
        ),
        # 6: Einsum inputs=7 outputs=1
        _node(
            'Einsum',
            ['packed', 'R', 'R', 'R', 'R', 'R', 'R'],
            ['output'],
            name='',
            domain='',
            equation='bkc,kr,kr,kr,kr,kr,lw->bcrw',
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
            _tensor('CASES', TensorProto.STRING, (28,), INIT_CASES_1),
            _tensor('R', TensorProto.INT64, (2, 30), INIT_R_2),
        ],
        value_info=[
            _vi('rng_f', TensorProto.FLOAT16, [1]),
            _vi('case_i', TensorProto.INT32, [1]),
            _vi('case_strings', TensorProto.STRING, [1]),
            _vi('pair_parts', TensorProto.STRING, [1, 2]),
            _vi('pair_lengths', TensorProto.INT64, [1]),
            _vi('parts', TensorProto.STRING, [1, 2, 10]),
            _vi('lengths', TensorProto.INT64, [1, 2]),
            _vi('packed', TensorProto.INT64, [1, 2, 10]),
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
