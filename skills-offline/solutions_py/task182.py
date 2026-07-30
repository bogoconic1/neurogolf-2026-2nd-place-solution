from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task182'
TASK_NUM = 182
KAGGLE = {'score': 19.696695, 'date': '2026-07-15'}
MEMORY_BYTES = 73
PARAMS = 128
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task182_l3_quant_tfidf'
OPSETS = [('', 21)]


# like3: FLOAT16[1, 2], 2 value(s)
INIT_LIKE3_0 = [0.0, 0.0]

# q_scale: FLOAT[1], 1 value(s)
INIT_Q_SCALE_1 = [1.0]

# q_zero: INT8[1], 1 value(s)
INIT_Q_ZERO_2 = [-7]

# const6_i8: INT8[1, 1], 1 value(s)
INIT_CONST6_I8_3 = [6]

# P: FLOAT[30, 2], 60 value(s)
INIT_P_4 = [1.1737592220306396, -0.7839196920394897, 1.1297953128814697, -0.9951030015945435, 1.1013176441192627,
 -0.9854886531829834, 1.138677954673767, -0.9856333136558533, 1.0352280139923096, -0.9135355353355408,
 1.0668344497680664, -0.8566502928733826, 1.2199286222457886, -0.7962329983711243, 1.3401336669921875,
 -0.639765202999115, 1.2314300537109375, -0.5013943314552307, 1.300642967224121, -0.3935335576534271,
 1.2497133016586304, -0.2729019522666931, 1.2598562240600586, -0.21387940645217896, 1.2114094495773315,
 0.030142007395625114, 1.2668097019195557, 0.2849178910255432, 1.1955777406692505, 0.4689685106277466,
 1.0981591939926147, 0.7019488215446472, 1.1415735483169556, 0.8256470561027527, 1.1959019899368286, 0.7643067836761475,
 1.1564345359802246, 0.732220470905304, 1.222077488899231, 0.675480842590332, 1.0, 1.105263113975525, 1.0,
 1.2105263471603394, 1.0, 1.3157894611358643, 1.0, 1.4210525751113892, 1.0, 1.5263158082962036, 1.0, 1.6315789222717285,
 1.0, 1.736842155456543, 1.0, 1.8421052694320679, 1.0, 1.9473683834075928, 1.0, 2.0526316165924072]

# U: FLOAT[2, 3, 2], 12 value(s)
INIT_U_5 = [0.8278571367263794, 0.15914785861968994, 0.6967900991439819, 0.2518336772918701, 0.13170845806598663,
 0.49055948853492737, -0.06618422269821167, 0.043644778430461884, -0.5182086229324341, 0.4966483414173126,
 0.989737868309021, -0.3053227663040161]

# T: FLOAT[2, 2, 3], 12 value(s)
INIT_T_6 = [-0.03452236205339432, -0.07415369898080826, 0.0704798772931099, 0.104596346616745, 0.04119383543729782,
 -0.09783101081848145, 0.19479955732822418, 0.24770915508270264, 0.05505494400858879, -0.11395829916000366,
 -0.13575206696987152, 0.18212759494781494]

# E: FLOAT[10, 3], 30 value(s)
INIT_E_7 = [0.17948411405086517, 3.1419122219085693, -0.8872620463371277, -7.484373092651367, 1.224892020225525,
 -6.957477569580078, -2.993823528289795, 1.022716999053955, -0.5248973965644836, -1.4795429706573486,
 1.8269637823104858, -3.608506917953491, -0.9087991714477539, 0.5151588320732117, -0.5069767832756042,
 -0.7909972667694092, 3.8745906352996826, 2.0042917728424072, -0.9146686792373657, 0.516957700252533,
 -0.5076860189437866, -0.9098920822143555, 0.5199909210205078, -0.5073150396347046, -0.9082538485527039,
 0.5337220430374146, -0.5077667832374573, -0.9141935706138611, 0.5209727883338928, -0.5078725814819336]

# V: FLOAT[3, 3], 9 value(s)
INIT_V_8 = [-1.5735561847686768, -1.5720369815826416, -1.0994219779968262, -1.517085313796997, -0.1051880493760109,
 0.35487622022628784, 1.0254487991333008, -0.265806645154953, 1.2305707931518555]


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
        # 0: RandomNormalLike inputs=1 outputs=1
        _node(
            'RandomNormalLike',
            ['like3'],
            ['rng3'],
            name='rng3',
            domain='',
            dtype=10,
            mean=30000.0,
            scale=10000.0,
            seed=1.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['rng3'],
            ['tokens'],
            name='tokens',
            domain='',
            to=6,
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['count0'],
            name='count0',
            domain='',
            max_gram_length=2,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 46],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[27409, 46015, 15010, 31747, 31192, 34581, 31889, 33949, 6301, 30441, 23355, 23637, 28287, 47312,
 22351, 37031, 30936, 42041, 46814, 22979, 31751, 32741, 21369, 41862, 43165, 36810, 24102, 32091,
 38604, 32944, 25331, 13312, 28682, 45766, 19527, 25580, 9958, 31804, 29959, 33286, 5815, 30147,
 34376, 27642, 6344, 49791, 27409, 46015, 33949, 6301, 30441, 23355, 23637, 28287, 47312, 22351,
 46814, 22979, 31751, 32741, 43165, 36810, 24102, 32091, 28682, 45766, 19527, 25580, 34376, 27642],
            weights=[2.6666667461395264, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['count1'],
            name='count1',
            domain='',
            max_gram_length=2,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 40],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[27409, 46015, 30441, 23355, 23637, 47312, 22351, 30936, 42041, 32610, 23080, 31751, 21369, 41862,
 43165, 36810, 39425, 38283, 24102, 32091, 38604, 32944, 28682, 45766, 19527, 25580, 39553, 29676,
 30102, 39197, 9958, 31804, 25783, 27590, 5815, 30147, 34376, 27642, 6344, 49791, 27409, 46015,
 32610, 23080, 21369, 41862, 43165, 36810, 39425, 38283, 24102, 32091, 38604, 32944, 28682, 45766,
 19527, 25580, 39553, 29676, 30102, 39197, 5815, 30147, 34376, 27642, 6344, 49791],
            weights=[4.666666507720947, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 4: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['count2'],
            name='count2',
            domain='',
            max_gram_length=2,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 48],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[27409, 15010, 31747, 31192, 34581, 31889, 33949, 30441, 23637, 47312, 22351, 37031, 31453, 30936,
 42041, 32610, 23080, 46814, 22979, 31751, 21369, 41862, 43165, 39425, 38283, 24102, 32091, 38604,
 32944, 25331, 28682, 22082, 19527, 25580, 39553, 30102, 39197, 9958, 29959, 33286, 25783, 27590,
 5815, 30147, 34376, 27642, 6344, 49791, 15010, 31747, 34581, 31889, 47312, 22351, 37031, 31453,
 30936, 42041, 32610, 23080, 46814, 22979, 21369, 41862, 39425, 38283, 24102, 32091, 38604, 32944,
 19527, 25580, 30102, 39197, 29959, 33286, 25783, 27590, 5815, 30147, 34376, 27642, 6344, 49791],
            weights=[2.6666667461395264, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 5: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['count3'],
            name='count3',
            domain='',
            max_gram_length=2,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 43],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[27409, 46015, 15010, 31747, 31192, 34581, 33949, 30441, 23637, 47312, 22351, 37031, 31453, 30936,
 42041, 32610, 23080, 46814, 31751, 39425, 38283, 24102, 38604, 32944, 25331, 28682, 22082, 20776,
 19527, 25580, 39553, 29676, 30102, 39197, 9958, 29959, 25783, 27590, 5815, 30147, 34376, 27642,
 6344, 27409, 46015, 47312, 22351, 37031, 31453, 30936, 42041, 32610, 23080, 39425, 38283, 22082,
 20776, 19527, 25580, 30102, 39197, 25783, 27590, 5815, 30147, 34376, 27642],
            weights=[4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 6: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['count4'],
            name='count4',
            domain='',
            max_gram_length=2,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 35],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[31192, 26979, 34581, 31889, 30441, 23355, 23637, 28287, 37031, 32610, 23080, 46814, 22979, 31751,
 32741, 21369, 41862, 43165, 36810, 39425, 38283, 24102, 25331, 13312, 28682, 45766, 39553, 29676,
 30102, 39197, 29959, 33286, 5815, 34376, 6344, 31192, 26979, 34581, 31889, 30441, 23355, 23637,
 28287, 32610, 23080, 31751, 32741, 39425, 38283, 39553, 29676, 30102, 39197, 29959, 33286],
            weights=[4.666666507720947, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 7: QuantizeLinear inputs=3 outputs=1
        _node(
            'QuantizeLinear',
            ['count0', 'q_scale', 'q_zero'],
            ['q0'],
            name='q0',
            domain='',
        ),
        # 8: QuantizeLinear inputs=3 outputs=1
        _node(
            'QuantizeLinear',
            ['count1', 'q_scale', 'q_zero'],
            ['q1'],
            name='q1',
            domain='',
        ),
        # 9: QuantizeLinear inputs=3 outputs=1
        _node(
            'QuantizeLinear',
            ['count2', 'q_scale', 'q_zero'],
            ['q2'],
            name='q2',
            domain='',
        ),
        # 10: QuantizeLinear inputs=3 outputs=1
        _node(
            'QuantizeLinear',
            ['count3', 'q_scale', 'q_zero'],
            ['q3'],
            name='q3',
            domain='',
        ),
        # 11: QuantizeLinear inputs=3 outputs=1
        _node(
            'QuantizeLinear',
            ['count4', 'q_scale', 'q_zero'],
            ['q4'],
            name='q4',
            domain='',
        ),
        # 12: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['q0', 'q1', 'q2'],
            ['row0_i8'],
            name='row0_i8',
            domain='',
            axis=1,
        ),
        # 13: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['q3', 'q4', 'const6_i8'],
            ['row1_i8'],
            name='row1_i8',
            domain='',
            axis=1,
        ),
        # 14: Concat inputs=2 outputs=1
        _node(
            'Concat',
            ['row0_i8', 'row1_i8'],
            ['descriptor_i8'],
            name='descriptor_i8',
            domain='',
            axis=0,
        ),
        # 15: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['descriptor_i8'],
            ['descriptor'],
            name='descriptor',
            domain='',
            to=1,
        ),
        # 16: Einsum inputs=19 outputs=1
        _node(
            'Einsum',
            ['descriptor', 'T', 'U', 'P', 'U', 'P', 'descriptor', 'U', 'P', 'U', 'P', 'descriptor', 'T', 'U',
             'T', 'E', 'E', 'V', 'input'],
            ['output'],
            name='output',
            domain='',
            equation='ka,kAB,tau,hu,qav,wv,tb,tbx,hx,dby,wy,pD,LCF,LDZ,tpr,cr,os,sr,nchw->nohw',
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
            _tensor('like3', TensorProto.FLOAT16, (1, 2), INIT_LIKE3_0),
            _tensor('q_scale', TensorProto.FLOAT, (1,), INIT_Q_SCALE_1),
            _tensor('q_zero', TensorProto.INT8, (1,), INIT_Q_ZERO_2),
            _tensor('const6_i8', TensorProto.INT8, (1, 1), INIT_CONST6_I8_3),
            _tensor('P', TensorProto.FLOAT, (30, 2), INIT_P_4),
            _tensor('U', TensorProto.FLOAT, (2, 3, 2), INIT_U_5),
            _tensor('T', TensorProto.FLOAT, (2, 2, 3), INIT_T_6),
            _tensor('E', TensorProto.FLOAT, (10, 3), INIT_E_7),
            _tensor('V', TensorProto.FLOAT, (3, 3), INIT_V_8),
        ],
        value_info=[
            _vi('rng3', TensorProto.FLOAT16, [1, 2]),
            _vi('tokens', TensorProto.INT32, [1, 2]),
            _vi('count0', TensorProto.FLOAT, [1, 1]),
            _vi('count1', TensorProto.FLOAT, [1, 1]),
            _vi('count2', TensorProto.FLOAT, [1, 1]),
            _vi('count3', TensorProto.FLOAT, [1, 1]),
            _vi('count4', TensorProto.FLOAT, [1, 1]),
            _vi('q0', TensorProto.INT8, [1, 1]),
            _vi('q1', TensorProto.INT8, [1, 1]),
            _vi('q2', TensorProto.INT8, [1, 1]),
            _vi('q3', TensorProto.INT8, [1, 1]),
            _vi('q4', TensorProto.INT8, [1, 1]),
            _vi('row0_i8', TensorProto.INT8, [1, 3]),
            _vi('row1_i8', TensorProto.INT8, [1, 3]),
            _vi('descriptor_i8', TensorProto.INT8, [2, 3]),
            _vi('descriptor', TensorProto.FLOAT, [2, 3]),
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
