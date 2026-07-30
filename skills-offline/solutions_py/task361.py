from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task361'
TASK_NUM = 361
KAGGLE = {'score': 19.142067, 'date': '2026-07-15'}
MEMORY_BYTES = 206
PARAMS = 144
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task361_v27_sum_mask'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# R: INT64[2, 30], 60 value(s)
INIT_R_1 = [512, 128, 32, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 256, 64,
 16, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# B: INT64[2, 10], 20 value(s)
INIT_B_2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# MASK_BITS: INT64[10], 10 value(s)
INIT_MASK_BITS_3 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# F2: INT64[6, 2, 2], 24 value(s)
INIT_F2_4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, -1, 1, 1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]

# CASES: STRING[29], 29 value(s)
INIT_CASES_5 = ['23732958485545183,22293228844953008,13621501920298496,2318009021,38696761207426224,755715415712337933',
 '496,18014398509481938,27021597764222846,5224919014374911280,2264851942663889858,695849615413813971',
 '181310566931890295,284197955647055,561248274828322,5507297372,17937512438643113,30032540344439060',
 '181423816629551276,8650738239487121,2115738522104013,4815771313426,150975915647505734,34313312525389871',
 '189301817442566425,45979881193253230,1169834558925557,1381021185,2305843007547362304,401924240',
 '23732958485545464,158377715143240,775987317417744,2318023218,108230107629032335,58066713186878320',
 '9024791440785504,180919598922246400,98285985788679594,1155177878732848140,2501189603754168573,5567002267211603968',
 '189301817442566166,13210061621323362,357926468321280,1381009993,65344634972864512,33047933132734304',
 '203969368072126559,134875148587455488,210522039502154507,1107603129425,384304864006895728,8839065513590947808',
 '348102340605116768,7973746363636092,1002299066013579,4584922329009,135106271509483008,38280739577743609',
 '472,24019198012641506,24019198012642969,5224919014374909998,2141394601027209519,755291027164279133',
 '564239702341714426,6973239253469128,410288488791533,10033214168024,51237054360455946,59247664716567424',
 '185830736543088917,45703564516916918,42116729416974401,2539327108333536,1439789284437098043,288402404501041650',
 '181423973395857448,8976829462696858,1470706209172183,5141734096867056,190037723937997560,24226785860784080',
 '194990863511715856,46643381224789320,32841586395354456,1139279892538,1537227816378368611,1537228815935773376',
 '23643902338662524,7755219509182464,66845420548696729,140738203130117,720493843154723247,2065660802563189376',
 '47465916971090014,1263836246856354,865216196463873,4636028656,108814267754479616,5042988521987472',
 '94598768892051730,58030951810902647,18183770129097601,2277299,1200959900630815948,368294369527343296',
 '144134979285156291,61375022165710847,9563182892737138,653027483696792581,450359962737049600,1049275470966492112',
 '239,26120877838748314,262109498312962970,1158732463590118528,136130056996419712,7926257147371383808',
 '166633186212708514,17264287243421379,21517035329047522,4982871080018,1277018168348141323,1880169663581942629',
 '284998921159180492,31472111730827456,6736241398968109,1136056774828,2228980777764711903,38430829562541281',
 '54122360365646076,8809287161741062,198237548441501502,2595766809537982247,2733960115435837760,2595763974856860768',
 '203934119258751285,14366675760928365,10503955584210020,37913611,2305843009167897088,11052540',
 '68781049387155764,4012623099717456,2525954120901003,8367167666776352001,27549674889712772,36330900971247341',
 '181446919258636647,1052963510419371,575818388471813,407937559,2001599738744135,2001599839967884',
 '181410622490018274,156288980818590915,148055837749805179,4403454459138,1220167479608557589,8003197952066629644',
 '189302916954194405,33431223704835170,71903807484490210,5437209422,144115184996533704,2001599834748607242',
 '167992491822284951,10467609199771536,17545580608633074,10139870889164682,1099839000242754926,1425693705663987256']


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
            ['case_strings'],
            name='',
            domain='',
            axis=0,
        ),
        # 3: StringSplit inputs=1 outputs=2
        _node(
            'StringSplit',
            ['case_strings'],
            ['parts', 'lengths'],
            name='',
            domain='',
            delimiter=',',
            maxsplit=5,
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
        # 5: ReduceSum inputs=1 outputs=1
        _node(
            'ReduceSum',
            ['packed'],
            ['mask_val'],
            name='',
            domain='',
            keepdims=1,
            noop_with_empty_axes=0,
        ),
        # 6: BitwiseAnd inputs=2 outputs=1
        _node(
            'BitwiseAnd',
            ['mask_val', 'MASK_BITS'],
            ['mask_c'],
            name='',
            domain='',
        ),
        # 7: Einsum inputs=12 outputs=1
        _node(
            'Einsum',
            ['packed', 'F2', 'B', 'B', 'mask_c', 'F2', 'R', 'R', 'R', 'R', 'R', 'R'],
            ['output'],
            name='',
            domain='',
            equation='bf,fde,dc,ec,bc,fku,kr,kr,kr,kr,kr,lw->bcrw',
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
            _tensor('R', TensorProto.INT64, (2, 30), INIT_R_1),
            _tensor('B', TensorProto.INT64, (2, 10), INIT_B_2),
            _tensor('MASK_BITS', TensorProto.INT64, (10,), INIT_MASK_BITS_3),
            _tensor('F2', TensorProto.INT64, (6, 2, 2), INIT_F2_4),
            _tensor('CASES', TensorProto.STRING, (29,), INIT_CASES_5),
        ],
        value_info=[
            _vi('rng_f', TensorProto.FLOAT16, [1]),
            _vi('case_i', TensorProto.INT32, [1]),
            _vi('case_strings', TensorProto.STRING, [1]),
            _vi('parts', TensorProto.STRING, [1, 6]),
            _vi('lengths', TensorProto.INT64, [1]),
            _vi('packed', TensorProto.INT64, [1, 6]),
            _vi('mask_val', TensorProto.INT64, [1, 1]),
            _vi('mask_c', TensorProto.INT64, [1, 10]),
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
