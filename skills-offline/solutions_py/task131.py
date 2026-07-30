from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task131'
TASK_NUM = 131
KAGGLE = {'score': 19.515203, 'date': '2026-07-14'}
MEMORY_BYTES = 66
PARAMS = 175
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task131_factorized_power_tie_3_8'
OPSETS = [('', 21)]


# baseexp: FLOAT[3], 3 value(s)
INIT_BASEEXP_0 = [1.0, 0.7788007855415344, 1.2840254306793213]

# ch2: FLOAT[10, 2], 20 value(s)
INIT_CH2_1 = [-1.0462241172790527, 309.88427734375, -0.0, -0.0, -1.0462241172790527, -0.0, -1.0462241172790527, 309.88427734375,
 -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -309.88427734375, -0.0, -0.0]

# CodesAG: FLOAT[2, 10], 20 value(s)
INIT_CODESAG_2 = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

# Coord: FLOAT[30, 3], 90 value(s)
INIT_COORD_3 = [1.0, 1.0, 1.0, 1.0, 1.2840255498886108, 0.7788007259368896, 1.0, 1.6487212181091309, 0.6065306663513184, 1.0,
 2.117000102996826, 0.4723665416240692, 1.0, 2.7182819843292236, 0.3678794205188751, 1.0, 3.490342855453491,
 0.2865047752857208, 1.0, 4.481688976287842, 0.22313015162944794, 1.0, 5.754601955413818, 0.17377394437789917, 1.0,
 7.3890557289123535, 0.1353352814912796, 1.0, 9.487735748291016, 0.1053992211818695, 1.0, 12.182493209838867,
 0.0820850059390068, 1.0, 15.642632484436035, 0.06392786651849747, 1.0, 20.08553695678711, 0.049787066876888275, 1.0,
 25.790340423583984, 0.03877420723438263, 1.0, 33.11545181274414, 0.03019738383591175, 1.0, 42.521080017089844,
 0.02351774461567402, 1.0, 54.598148345947266, 0.018315639346837997, 1.0, 70.10541534423828, 0.014264233410358429, 1.0,
 70.10541534423828, 0.014264233410358429, 1.0, 70.10541534423828, 0.014264233410358429, 1.0, 70.10541534423828,
 0.014264233410358429, 1.0, 70.10541534423828, 0.014264233410358429, 1.0, 70.10541534423828, 0.014264233410358429, 1.0,
 70.10541534423828, 0.014264233410358429, 1.0, 70.10541534423828, 0.014264233410358429, 1.0, 70.10541534423828,
 0.014264233410358429, 1.0, 70.10541534423828, 0.014264233410358429, 1.0, 70.10541534423828, 0.014264233410358429, 1.0,
 70.10541534423828, 0.014264233410358429, 1.0, 70.10541534423828, 0.014264233410358429]

# Swap: FLOAT[3, 3], 9 value(s)
INIT_SWAP_4 = [1.0, 0.0, 0.0, 0.0, 0.0, 15.642630577087402, 0.0, 0.06392785161733627, 0.0]

# Bv: FLOAT[2, 2, 2], 8 value(s)
INIT_BV_5 = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]

# CoeffCommon: FLOAT[2, 2, 3], 12 value(s)
INIT_COEFFCOMMON_6 = [1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 2.062826156616211, 0.0, 0.0]

# TfP: FLOAT[2, 2, 2], 8 value(s)
INIT_TFP_7 = [1.0, -3.2360785553464666e-06, -0.7532771229743958, -0.0028243192937225103, -3815.7490234375, 0.0011811066651716828,
 -8.043442726135254, 1.0290781259536743]

# rng_tpl: FLOAT16[1], 1 value(s)
INIT_RNG_TPL_8 = [0.0]

# qA: FLOAT[2, 2], 4 value(s)
INIT_QA_9 = [70.21578216552734, -103211144.0, 421.13067626953125, -57425.375]


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
            ['rng_tpl'],
            ['rng_f'],
            name='',
            domain='',
            dtype=10,
            high=60000.0,
            low=0.0,
            seed=123.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['rng_f'],
            ['tok_scalar'],
            name='',
            domain='',
            to=6,
        ),
        # 2: Concat inputs=5 outputs=1
        _node(
            'Concat',
            ['tok_scalar', 'tok_scalar', 'tok_scalar', 'tok_scalar', 'tok_scalar'],
            ['tok_seq'],
            name='',
            domain='',
            axis=0,
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok_seq'],
            ['shift_f'],
            name='',
            domain='',
            max_gram_length=5,
            max_skip_count=3,
            min_gram_length=1,
            mode='TF',
            ngram_counts=[0, 21, 61, 109, 169],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[57, 10748, 24837, 31822, 55918, 33207, 58033, 10245, 53292, 15142, 34546, 9187, 56812, 9607, 13295,
 21046, 26647, 26621, 48734, 25921, 10075, 57, 57, 10748, 10748, 31822, 31822, 55918, 55918, 33207,
 33207, 58033, 58033, 10245, 10245, 53292, 53292, 15142, 15142, 34546, 34546, 9187, 9187, 56812,
 56812, 9607, 9607, 13295, 13295, 21046, 21046, 26621, 26621, 48734, 48734, 18700, 18700, 25921,
 25921, 10075, 10075, 10748, 10748, 10748, 56367, 56367, 56367, 47189, 47189, 47189, 30021, 30021,
 30021, 33207, 33207, 33207, 58033, 58033, 58033, 53292, 53292, 53292, 34546, 34546, 34546, 9187,
 9187, 9187, 56812, 56812, 56812, 9607, 9607, 9607, 13295, 13295, 13295, 28382, 28382, 28382, 26621,
 26621, 26621, 48734, 48734, 48734, 18700, 18700, 18700, 57, 57, 57, 57, 10748, 10748, 10748, 10748,
 31822, 31822, 31822, 31822, 55918, 55918, 55918, 55918, 33207, 33207, 33207, 33207, 10245, 10245,
 10245, 10245, 53292, 53292, 53292, 53292, 12524, 12524, 12524, 12524, 15142, 15142, 15142, 15142,
 33082, 33082, 33082, 33082, 9607, 9607, 9607, 9607, 21046, 21046, 21046, 21046, 18700, 18700,
 18700, 18700, 25921, 25921, 25921, 25921, 10075, 10075, 10075, 10075, 31822, 31822, 31822, 31822,
 31822, 33207, 33207, 33207, 33207, 33207, 58033, 58033, 58033, 58033, 58033, 12524, 12524, 12524,
 12524, 12524, 15142, 15142, 15142, 15142, 15142, 56812, 56812, 56812, 56812, 56812, 9607, 9607,
 9607, 9607, 9607, 13295, 13295, 13295, 13295, 13295, 21046, 21046, 21046, 21046, 21046, 26621,
 26621, 26621, 26621, 26621, 48734, 48734, 48734, 48734, 48734, 10075, 10075, 10075, 10075, 10075],
        ),
        # 4: Pow inputs=2 outputs=1
        _node(
            'Pow',
            ['baseexp', 'shift_f'],
            ['state_shift_exp'],
            name='',
            domain='',
        ),
        # 5: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok_seq'],
            ['cyan_f'],
            name='',
            domain='',
            max_gram_length=5,
            max_skip_count=3,
            min_gram_length=1,
            mode='TF',
            ngram_counts=[0, 23, 29, 62, 126],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[57, 56367, 24837, 31822, 47189, 30021, 58033, 10245, 53292, 12524, 15142, 34546, 9187, 33082,
 56812, 13295, 21046, 28382, 26647, 26621, 4711, 48734, 25921, 10748, 10748, 33207, 33207, 9607,
 9607, 55918, 55918, 55918, 47189, 47189, 47189, 30021, 30021, 30021, 58033, 58033, 58033, 53292,
 53292, 53292, 56812, 56812, 56812, 13295, 13295, 13295, 28382, 28382, 28382, 26621, 26621, 26621,
 18700, 18700, 18700, 10075, 10075, 10075, 56367, 56367, 56367, 56367, 24837, 24837, 24837, 24837,
 31822, 31822, 31822, 31822, 55918, 55918, 55918, 55918, 10245, 10245, 10245, 10245, 12524, 12524,
 12524, 12524, 15142, 15142, 15142, 15142, 34546, 34546, 34546, 34546, 9187, 9187, 9187, 9187,
 33082, 33082, 33082, 33082, 13295, 13295, 13295, 13295, 21046, 21046, 21046, 21046, 26647, 26647,
 26647, 26647, 48734, 48734, 48734, 48734, 25921, 25921, 25921, 25921, 10075, 10075, 10075, 10075,
 24837, 24837, 24837, 24837, 24837, 31822, 31822, 31822, 31822, 31822, 15142, 15142, 15142, 15142,
 15142, 34546, 34546, 34546, 34546, 34546, 9187, 9187, 9187, 9187, 9187, 21046, 21046, 21046, 21046,
 21046, 26647, 26647, 26647, 26647, 26647, 48734, 48734, 48734, 48734, 48734, 25921, 25921, 25921,
 25921, 25921],
        ),
        # 6: Pow inputs=2 outputs=1
        _node(
            'Pow',
            ['baseexp', 'cyan_f'],
            ['state_cyan_exp'],
            name='',
            domain='',
        ),
        # 7: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok_seq'],
            ['state_vh'],
            name='',
            domain='',
            max_gram_length=5,
            max_skip_count=3,
            min_gram_length=1,
            mode='TF',
            ngram_counts=[0, 0, 0, 0, 0],
            ngram_indexes=[0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            pool_int64s=[57, 57, 57, 57, 57, 10748, 10748, 10748, 10748, 10748, 56367, 56367, 56367, 56367, 56367, 24837,
 24837, 24837, 24837, 24837, 31822, 31822, 31822, 31822, 31822, 55918, 55918, 55918, 55918, 55918,
 47189, 47189, 47189, 47189, 47189, 30021, 30021, 30021, 30021, 30021, 33207, 33207, 33207, 33207,
 33207, 58033, 58033, 58033, 58033, 58033, 10245, 10245, 10245, 10245, 10245, 53292, 53292, 53292,
 53292, 53292, 12524, 12524, 12524, 12524, 12524, 15142, 15142, 15142, 15142, 15142, 34546, 34546,
 34546, 34546, 34546, 9187, 9187, 9187, 9187, 9187, 33082, 33082, 33082, 33082, 33082, 56812, 56812,
 56812, 56812, 56812, 9607, 9607, 9607, 9607, 9607, 13295, 13295, 13295, 13295, 13295, 21046, 21046,
 21046, 21046, 21046, 28382, 28382, 28382, 28382, 28382, 26647, 26647, 26647, 26647, 26647, 26621,
 26621, 26621, 26621, 26621, 4711, 4711, 4711, 4711, 4711, 48734, 48734, 48734, 48734, 48734, 18700,
 18700, 18700, 18700, 18700, 25921, 25921, 25921, 25921, 25921, 10075, 10075, 10075, 10075, 10075],
        ),
        # 8: Einsum inputs=152 outputs=1
        _node(
            'Einsum',
            ['CoeffCommon', 'Swap', 'Coord', 'Coord', 'state_shift_exp', 'TfP', 'TfP', 'TfP', 'CoeffCommon',
             'Coord', 'Swap', 'Coord', 'state_shift_exp', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP',
             'TfP', 'CoeffCommon', 'Coord', 'Swap', 'Coord', 'state_shift_exp', 'TfP', 'TfP', 'TfP', 'TfP',
             'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'CoeffCommon', 'Coord',
             'Swap', 'Coord', 'state_shift_exp', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP',
             'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'CoeffCommon', 'Coord', 'Swap', 'Coord', 'state_shift_exp',
             'CodesAG', 'input', 'CoeffCommon', 'Coord', 'Swap', 'Coord', 'state_shift_exp', 'TfP', 'TfP',
             'TfP', 'CoeffCommon', 'Coord', 'Swap', 'Coord', 'state_shift_exp', 'TfP', 'TfP', 'TfP', 'TfP',
             'TfP', 'TfP', 'TfP', 'TfP', 'CoeffCommon', 'Coord', 'Swap', 'Coord', 'state_shift_exp', 'TfP',
             'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP',
             'CoeffCommon', 'Coord', 'Swap', 'Coord', 'state_shift_exp', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP',
             'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'TfP', 'CoeffCommon', 'Coord', 'Swap',
             'Coord', 'state_shift_exp', 'CodesAG', 'input', 'Bv', 'Bv', 'Bv', 'state_vh', 'Coord',
             'state_cyan_exp', 'CoeffCommon', 'Bv', 'Bv', 'Bv', 'CoeffCommon', 'Coord', 'state_cyan_exp',
             'input', 'ch2', 'CodesAG', 'ch2', 'TfP', 'Bv', 'qA'],
            ['output'],
            name='',
            domain='',
            equation='bqA,AF,jF,cA,A,ddd,ddd,ddd,dqB,cB,BG,jG,B,eee,eee,eee,eee,eee,eee,eee,eee,eqC,cC,CH,jH,C,fff,fff,fff,fff,fff,fff,fff,fff,fff,fff,fff,fff,fff,fff,fqD,cD,DI,jI,D,ggg,ggg,ggg,ggg,ggg,ggg,ggg,ggg,ggg,ggg,ggg,ggg,ggg,ggg,gqE,cE,EJ,jJ,E,qu,...urj,phK,rK,KR,aR,K,sss,sss,sss,shM,rM,MS,aS,M,ttt,ttt,ttt,ttt,ttt,ttt,ttt,ttt,thN,rN,NT,aT,N,www,www,www,www,www,www,www,www,www,www,www,www,www,www,whP,rP,PU,aU,P,zzz,zzz,zzz,zzz,zzz,zzz,zzz,zzz,zzz,zzz,zzz,zzz,zzz,zzz,zhQ,rQ,QV,aV,Q,hv,...vac,WOq,WOn,Wnh,O,cx,x,mkx,XOk,XOk,Xkl,mly,ry,y,...irc,iL,Yo,oZ,WXZ,LYL,XY->...orc',
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
            _tensor('baseexp', TensorProto.FLOAT, (3,), INIT_BASEEXP_0),
            _tensor('ch2', TensorProto.FLOAT, (10, 2), INIT_CH2_1),
            _tensor('CodesAG', TensorProto.FLOAT, (2, 10), INIT_CODESAG_2),
            _tensor('Coord', TensorProto.FLOAT, (30, 3), INIT_COORD_3),
            _tensor('Swap', TensorProto.FLOAT, (3, 3), INIT_SWAP_4),
            _tensor('Bv', TensorProto.FLOAT, (2, 2, 2), INIT_BV_5),
            _tensor('CoeffCommon', TensorProto.FLOAT, (2, 2, 3), INIT_COEFFCOMMON_6),
            _tensor('TfP', TensorProto.FLOAT, (2, 2, 2), INIT_TFP_7),
            _tensor('rng_tpl', TensorProto.FLOAT16, (1,), INIT_RNG_TPL_8),
            _tensor('qA', TensorProto.FLOAT, (2, 2), INIT_QA_9),
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
