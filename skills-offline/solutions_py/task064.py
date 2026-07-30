from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task064'
TASK_NUM = 64
KAGGLE = {'score': 19.852506, 'date': '2026-07-13'}
MEMORY_BYTES = 104
PARAMS = 68
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 7
PRODUCER_NAME = 'OpenAI-neurogolf'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task064_memorized_qpresent_no_counts_noe0'
OPSETS = [('', 12)]


# Q: FLOAT[2, 30], 60 value(s)
INIT_Q_0 = [1.972959041595459, 1.0199668407440186, 0.5272954702377319, 0.27259761095046997, 0.14092564582824707,
 0.07285477966070175, 0.037663962692022324, 0.01947125792503357, 0.010066118091344833, 0.00520391296595335,
 0.0026902833487838507, 0.0013908041873946786, 0.0007190084434114397, 0.0003717080398928374, 0.0001921630755532533,
 9.93431531242095e-05, 5.1357743359403685e-05, 2.65505732386373e-05, 1.3725933968089521e-05, 7.0959399636194576e-06,
 3.6684102724393597e-06, 1.8964697119372431e-06, 9.804239198274445e-07, 5.068528139418049e-07, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 5.068528139418049e-07, 9.804239198274445e-07, 1.8964697119372431e-06, 3.6684102724393597e-06,
 7.0959399636194576e-06, 1.3725933968089521e-05, 2.65505732386373e-05, 5.1357743359403685e-05, 9.93431531242095e-05,
 0.0001921630755532533, 0.0003717080398928374, 0.0007190084434114397, 0.0013908041873946786, 0.0026902833487838507,
 0.00520391296595335, 0.010066118091344833, 0.01947125792503357, 0.037663962692022324, 0.07285477966070175,
 0.14092564582824707, 0.27259761095046997, 0.5272954702377319, 1.0199668407440186, 1.972959041595459, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0]

# T: FLOAT[2, 2, 2], 8 value(s)
INIT_T_1 = [-1.0, 0.0, 0.0, -2.111837148666382, 1.0, -1.2022114992141724, 2.9687259197235107, 0.0]


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
        # 0: Multinomial inputs=1 outputs=1
        _node(
            'Multinomial',
            ['Q'],
            ['tok'],
            name='',
            domain='',
            sample_size=3,
            seed=999.0,
            dtype=6,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['q'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=0,
            min_gram_length=2,
            mode='TF',
            ngram_counts=[0, 0, 120],
            ngram_indexes=[3, 4, 1, 2, 2, 1, 4, 1, 1, 1, 5, 1, 2, 4, 4, 3, 2, 5, 4, 7, 1, 2, 6, 7, 2, 2, 5, 3, 4, 4, 5, 5, 4,
 8, 1, 1, 4, 3, 3, 4, 6, 4, 4, 4, 2, 5, 3, 6, 2, 1, 7, 1, 4, 2, 2, 1, 3, 1, 7, 2, 4, 8, 8, 8, 4, 2,
 5, 4, 8, 2, 7, 5, 6, 6, 9, 9, 5, 8, 7, 9, 2, 8, 9, 9, 4, 3, 6, 6, 9, 7, 7, 6, 8, 9, 7, 4, 9, 9, 8,
 8, 7, 6, 7, 6, 5, 9, 6, 8, 4, 2, 9, 7, 9, 4, 8, 8, 9, 3, 8, 7],
            pool_int64s=[6, 1, 7, 23, 11, 0, 3, 19, 2, 9, 17, 16, 16, 28, 22, 22, 29, 2, 13, 23, 0, 18, 23, 23, 26, 29, 17,
 9, 26, 17, 0, 23, 17, 29, 5, 3, 29, 12, 7, 20, 25, 0, 8, 15, 16, 4, 23, 17, 4, 4, 5, 21, 14, 2, 1,
 23, 26, 20, 28, 28, 19, 0, 0, 29, 19, 21, 19, 18, 0, 26, 0, 27, 0, 8, 5, 17, 18, 0, 14, 21, 2, 23,
 17, 23, 14, 3, 21, 25, 27, 4, 22, 27, 0, 25, 9, 26, 18, 13, 20, 3, 1, 22, 17, 12, 2, 11, 14, 10, 0,
 11, 24, 16, 1, 16, 25, 23, 28, 17, 9, 18, 6, 1, 19, 7, 23, 25, 11, 0, 17, 23, 3, 19, 6, 2, 9, 17,
 16, 26, 16, 28, 13, 22, 22, 23, 29, 2, 15, 13, 23, 19, 0, 18, 9, 22, 23, 23, 26, 29, 0, 17, 9, 23,
 26, 17, 17, 0, 23, 5, 17, 29, 8, 5, 3, 0, 29, 12, 23, 7, 20, 11, 3, 25, 0, 23, 8, 15, 16, 4, 5, 23,
 17, 0, 4, 4, 0, 5, 21, 22, 14, 2, 28, 1, 23, 25, 26, 20, 0, 28, 28, 24, 24, 19, 0, 0, 29, 17, 19,
 21, 0, 24, 19, 18, 0, 26, 14, 23, 0, 27, 23, 0, 8, 5, 17, 14, 18, 0, 0, 14, 21, 5, 2, 23, 3, 17,
 23, 25, 14, 3, 25, 21, 25, 22, 27, 4, 7, 22, 27, 23, 0, 25, 3, 9, 26, 23, 18, 13, 26, 20, 3, 29, 1,
 22, 15, 17, 12, 22, 6, 2, 11, 14, 10, 23, 0, 11, 7, 24, 16, 17, 1, 16, 12, 25, 23, 8, 28, 17, 0, 9,
 18, 11],
        ),
        # 2: Einsum inputs=110 outputs=1
        _node(
            'Einsum',
            ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'q',
             'q', 'Q', 'Q', 'input', 'input', 'q', 'q', 'T', 'T', 'Q', 'Q', 'T', 'T', 'T', 'T', 'q', 'q',
             'input', 'Q', 'Q', 'T', 'T', 'T', 'T', 'T', 'Q', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T',
             'T', 'T', 'T', 'T', 'q', 'q', 'input', 'Q', 'Q', 'Q', 'Q', 'input', 'q', 'q', 'T', 'T', 'T', 'T',
             'T', 'T', 'T', 'T', 'q', 'q', 'input', 'Q', 'Q', 'T', 'T', 'Q', 'Q', 'Q', 'input', 'q', 'q', 'T',
             'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'q', 'q'],
            ['output'],
            name='',
            domain='',
            equation='abb,dba,baa,bfa,ffa,bna,bna,nbp,fpa,nbp,pfo,mbo,mbo,bma,bma,kff,kka,lff,kla,ki,li,nj,aj,...ihj,...xhy,Ax,zx,Aza,zff,ah,bh,sff,ssa,sta,tff,sq,tq,...qhr,br,vr,fva,vfu,bFa,FbH,FbH,Fh,bFa,bGa,bGa,GbI,GbI,IgH,gIa,fgg,fYY,bYa,gYY,bga,Rgg,SRa,RP,SP,...PQw,uw,mw,GC,aC,...BCw,EB,DB,DEa,Egg,DDa,Dgg,LLa,Lgg,LMa,Mgg,LJ,MJ,...JKw,bK,OK,gOa,OgN,Nh,aw,bw,...Thw,UT,VT,YbU,VZa,ZUU,aZZ,Zaa,WbY,bWa,WbY,bWa,Zaa,Zaa,YZa,XZa,eYY,eea,eea,eXX,Xc,Wc->...chw',
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
            _tensor('Q', TensorProto.FLOAT, (2, 30), INIT_Q_0),
            _tensor('T', TensorProto.FLOAT, (2, 2, 2), INIT_T_1),
        ],
        value_info=[
            _vi('tok', TensorProto.INT32, [2, 3]),
            _vi('q', TensorProto.FLOAT, [2, 10]),
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
