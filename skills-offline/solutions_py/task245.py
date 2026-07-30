from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task245'
TASK_NUM = 245
KAGGLE = {'score': 19.5839, 'date': '2026-07-15'}
MEMORY_BYTES = 102
PARAMS = 123
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task245_absorbR_hp10_n5'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# CASES: STRING[32], 32 value(s)
INIT_CASES_1 = ['322140064727157,12217529062682108,51320474478452,25413919273304,14944321666464',
 '11206424412180480,61251987838185472,35344539348992,20054064386048,25828736542720',
 '17188321237682982,68520643559663947,56170758621873,131116174248577,14944321666464',
 '56419931092776826,62180484101219462,58637802200423,1379861410853,7244520134560',
 '71453653292708192,27146696430904025,15200789631483,105214020099293,11724661857697',
 '44144917273016879,8248756872014630,11662172304796,42886562336916,5428807087744',
 '55508659185318176,42538746869812704,5323024930452,23876492558520,32845463202304',
 '64950545314040257,67323115280136681,50233221851000,40007346291880,11724661857697',
 '34689270685499545,40471776936129895,68485741648896,118770305635328,14944321666464',
 '70606770559245256,3398174969193976,54356095481244,43181364498580,30506554315776',
 '47598474872501646,20902172353766770,8922783809536,95125586051072,14944321666464',
 '20206069887915518,54363097411954494,52422767490460,21398112194708,5428807087744',
 '33544411277713178,51575916319055647,23727326717696,96451652232448,4306049358810',
 '40701336048027667,67733028486629869,15483810830784,65250677033280,16464238845648',
 '54534060308520271,42611189454639197,40120348983655,102344263429157,21214446111696',
 '9141544962028965,60824417795483739,2101339125415,25908148232101,30506554315776', '0,0,0,0,0',
 '5889893010480025,27022528620234911,38140238335990,90336769999290,14944321666464',
 '12615359537597737,43689798605551859,39714035507719,81945349421541,4306049358810',
 '23711733522948386,64642894887812318,39572115373733,62299096333102,25824305467240',
 '20311039112330507,4267354635501045,18188003331431,113596473578533,4762197951668',
 '25979365993861504,32218421176640924,66637927468391,72055609758757,14944321666464',
 '63730878592165106,21837973630755710,58978358722360,135796823611688,7244520134560',
 '4483756326264008,2685566506829048,39960427642533,31513878379310,25824305467240',
 '46004469013507569,42830408097277376,18064436435392,87977586477376,32913377644672',
 '25426088274488437,55443104636980960,26917230856039,133562448335141,11724661857697',
 '37814998118541231,44968884193726145,12878354055168,99227104444416,25824305467240',
 '41784741735830503,37793942739605674,64387910155941,140075759287086,5428807087744',
 '32984514082407368,7971715340815871,5244430218919,49729894145957,25824305467240',
 '60483115538380791,35900873782995689,56635780340430,32230976530506,4306049358810',
 '41620993163481405,3199855239467999,3639884035850,122506573848476,7244520134560',
 '6045369925504997,769818686766107,44572453793792,109196919066624,14944321666464']

# H: INT64[5, 2], 10 value(s)
INIT_H_2 = [9, 1, 729, 1, 3, 1, 27, 1, 243, 1]

# C: INT64[2, 10], 20 value(s)
INIT_C_3 = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, -14348907, 0, -205891132094649, -729, 0, 0, 0, 0, 0, 0]

# P: INT64[2, 30], 60 value(s)
INIT_P_4 = [1, 2, 4, 8, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -179, -358, -716, -1432,
 -2864, 32, 64, 128, 256, 512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


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
            ['case_f'],
            name='',
            domain='',
            dtype=10,
            high=3.9803617000579834,
            low=-29.133535385131836,
            seed=26998238.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['case_f'],
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
            maxsplit=4,
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
        # 5: Einsum inputs=56 outputs=1
        _node(
            'Einsum',
            ['packed', 'H', 'C', 'H', 'H', 'C', 'H', 'H', 'H', 'C', 'H', 'H', 'H', 'H', 'H', 'C', 'H', 'H', 'H',
             'H', 'H', 'H', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C',
             'H', 'P', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'P', 'P', 'P', 'P', 'P'],
            ['output'],
            name='',
            domain='',
            equation='bf,fa,ak,fc,fc,ck,fd,fd,fd,dk,fe,fe,fe,fe,fe,ek,fg,fg,fg,fg,fg,fg,gk,fi,fi,fi,fi,fi,fi,fi,fi,fi,fi,fi,fi,fi,fi,fi,ik,fq,qr,fs,fs,fs,fs,fs,fs,fs,fs,fs,fs,sw,sw,sw,sw,sw->bkrw',
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
            _tensor('CASES', TensorProto.STRING, (32,), INIT_CASES_1),
            _tensor('H', TensorProto.INT64, (5, 2), INIT_H_2),
            _tensor('C', TensorProto.INT64, (2, 10), INIT_C_3),
            _tensor('P', TensorProto.INT64, (2, 30), INIT_P_4),
        ],
        value_info=[
            _vi('case_f', TensorProto.FLOAT16, [1]),
            _vi('case_i', TensorProto.INT32, [1]),
            _vi('case_string', TensorProto.STRING, [1]),
            _vi('parts', TensorProto.STRING, [1, 5]),
            _vi('lengths', TensorProto.INT64, [1]),
            _vi('packed', TensorProto.INT64, [1, 5]),
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
