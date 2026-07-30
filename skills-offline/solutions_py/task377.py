from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task377'
TASK_NUM = 377
KAGGLE = {'score': 19.519361, 'date': '2026-07-15'}
MEMORY_BYTES = 190
PARAMS = 50
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task377_tfidf_count_ids'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1, 1], 1 value(s)
INIT_LIKE_0 = [0.0]

# SENT: INT32[1, 1], 1 value(s)
INIT_SENT_1 = [12345]

# MASKS: INT64[18], 18 value(s)
INIT_MASKS_2 = [-554342976399081472, -2811058599835467776, -1722876665857950115, -4292654711701194705, 565278670702772718,
 89769200539326360, -1436534732038476274, 1937100097171731061, 288230376151711744, 302110635860150433,
 -8250043085495468032, -9090401242699530240, 8140700761315914497, 4229950269186472794, 1085920944980238112,
 2801829042294622232, 2613301710318313150, 539392508527840432]

# V: INT64[30], 30 value(s)
INIT_V_3 = [-1703936, 13, 37748736, -822083584, 172, 46170898432, 328, 6016, -90177536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0]


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
            high=11.34144401550293,
            low=-20.294286727905273,
            seed=891039552.0,
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
        # 2: Concat inputs=6 outputs=1
        _node(
            'Concat',
            ['SENT', 'case_i', 'case_i', 'case_i', 'case_i', 'SENT'],
            ['key_seq'],
            name='',
            domain='',
            axis=1,
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key_seq'],
            ['ids_f'],
            name='',
            domain='',
            max_gram_length=6,
            max_skip_count=3,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 30, 210, 423, 587, 642],
            ngram_indexes=[2, 7, 2, 7, 7, 4, 1, 5, 9, 6, 4, 3, 4, 8, 8, 3, 4, 1, 2, 2, 2, 3, 4, 3, 5, 5, 7, 8, 6, 3, 1, 2, 2,
 7, 2, 2, 7, 7, 7, 7, 8, 4, 6, 1, 3, 5, 3, 9, 3, 6, 4, 4, 9, 3, 6, 4, 6, 8, 3, 8, 5, 3, 3, 4, 3, 1,
 8, 2, 4, 2, 7, 2, 5, 3, 5, 4, 3, 3, 6, 5, 6, 5, 8, 7, 3, 8, 8, 6, 4, 3, 2, 7, 3, 2, 1, 4, 1, 5, 9,
 6, 8, 3, 4, 3, 8, 3, 4, 1, 2, 2, 2, 3, 4, 2, 5, 5, 7, 8, 6, 3, 1, 1, 2, 2, 3, 8, 2, 6, 7, 2, 8, 8,
 6, 6, 3, 3, 3, 3, 3, 3, 4, 9, 9, 6, 6, 6, 3, 3, 3, 5, 5, 3, 3, 3, 3, 8, 8, 4, 4, 7, 7, 5, 5, 5, 5,
 2, 1, 6, 6, 6, 6, 8, 8, 3, 3, 8, 8, 4, 4, 6, 2, 1, 4, 7, 7, 6, 3, 3, 8, 1, 3, 2, 3, 6, 3, 1, 1, 6,
 9, 3, 6, 3, 4, 8, 3, 3, 5, 1, 3, 2, 8, 2, 7, 3, 5, 1, 8, 6, 8, 8, 3, 2, 1, 4, 7, 7, 6, 3, 3, 8, 8,
 3, 1, 2, 1, 4, 4, 7, 6, 3, 3, 8, 3],
            pool_int64s=[-19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3,
 4, 5, 7, 8, 9, 10, 11, -19, -19, -19, 12345, -18, -18, -18, 12345, -17, -17, -17, 12345, -16, -16,
 -16, 12345, -15, -15, -15, 12345, -14, -14, -14, 12345, -13, -13, -13, 12345, -12, -12, -12, 12345,
 -11, -11, -11, 12345, -10, -10, -10, 12345, -9, -9, -9, 12345, -8, -8, -8, 12345, -7, -7, -7,
 12345, -6, -6, -6, 12345, -5, -5, -5, 12345, -4, -4, -4, 12345, -3, -3, -3, 12345, -2, -2, -2,
 12345, -1, -1, -1, 12345, 0, 0, 0, 12345, 1, 1, 1, 12345, 2, 2, 2, 12345, 3, 3, 3, 12345, 4, 4, 4,
 12345, 5, 5, 5, 12345, 7, 7, 7, 12345, 8, 8, 8, 12345, 9, 9, 9, 12345, 10, 10, 10, 12345, 11, 11,
 11, 12345, 12345, -19, 12345, -18, 12345, -17, 12345, -16, 12345, -15, 12345, -14, 12345, -13,
 12345, -12, 12345, -11, 12345, -10, 12345, -9, 12345, -8, 12345, -7, 12345, -6, 12345, -5, 12345,
 -4, 12345, -3, 12345, -2, 12345, -1, 12345, 0, 12345, 1, 12345, 2, 12345, 3, 12345, 4, 12345, 5,
 12345, 7, 12345, 8, 12345, 9, 12345, 10, 12345, 11, -19, -19, -19, -19, -19, 12345, -18, -18, -18,
 -18, -18, 12345, -17, -17, -17, -17, -17, 12345, -16, -16, -16, -16, -16, 12345, -15, -15, -15,
 -15, -15, 12345, -14, -14, -14, -14, -14, 12345, -13, -13, -13, -13, -13, 12345, -12, -12, -12,
 -12, -12, 12345, -11, -11, -11, -11, -11, 12345, -10, -10, -10, -10, -10, 12345, -9, -9, -9, -8,
 -8, -8, -8, -8, 12345, -7, -7, -7, -7, -7, 12345, -6, -6, -6, -6, -6, 12345, -5, -5, -5, -5, -5,
 12345, -4, -4, -4, -4, -4, 12345, -3, -3, -3, -3, -3, 12345, -2, -2, -2, -2, -2, 12345, -1, -1, -1,
 -1, -1, 12345, 0, 0, 0, 0, 0, 12345, 1, 1, 1, 1, 1, 12345, 2, 2, 2, 2, 2, 12345, 3, 3, 3, 3, 3,
 12345, 4, 4, 4, 4, 4, 12345, 5, 5, 5, 5, 5, 12345, 7, 7, 7, 7, 7, 12345, 8, 8, 8, 8, 8, 12345, 9,
 9, 9, 9, 9, 12345, 10, 10, 10, 10, 10, 12345, 11, 11, 11, 11, 11, 12345, 12345, -16, -16, 12345,
 -13, -13, 12345, -11, -11, 12345, -10, -10, 12345, -6, -6, 12345, -4, -4, 12345, -2, -2, 12345, -1,
 -1, 12345, 1, 1, 12345, 2, 2, 12345, 4, 4, 12345, 10, 10, -17, -17, -17, -17, -17, -17, -17, 12345,
 -16, -16, -16, -16, -16, -16, -16, 12345, -15, -15, -15, -15, -13, -13, -13, -13, -13, -13, -13,
 12345, -11, -11, -11, -11, -11, -11, -11, 12345, -10, -10, -10, -10, -10, -10, -10, 12345, -9, -9,
 -9, -9, -6, -6, -6, -6, -6, -6, -6, 12345, -4, -4, -4, -4, -4, -4, -4, 12345, -2, -2, -2, -2, -2,
 -2, -2, 12345, -1, -1, -1, -1, -1, -1, -1, 12345, 1, 1, 1, 1, 1, 1, 1, 12345, 2, 2, 2, 2, 2, 2, 2,
 12345, 4, 4, 4, 4, 4, 4, 4, 12345, 10, 10, 10, 10, 10, 10, 10, 12345, 12345, -17, -17, -17, 12345,
 -16, -16, -16, 12345, -13, -13, -13, 12345, -11, -11, -11, 12345, -10, -10, -10, 12345, -6, -6, -6,
 12345, -4, -4, -4, 12345, -2, -2, -2, 12345, -1, -1, -1, 12345, 1, 1, 1, 12345, 2, 2, 2, 12345, 4,
 4, 4, 12345, 10, 10, 10, -17, -17, -17, -17, 12345, -13, -13, -13, -13, 12345, -11, -11, -11, -11,
 12345, -10, -10, -10, -10, 12345, -6, -6, -6, -6, 12345, -4, -4, -4, -4, 12345, -2, -2, -2, -2,
 12345, -1, -1, -1, -1, 12345, 1, 1, 1, 1, 12345, 2, 2, 2, 2, 12345, 10, 10, 10, 10, 12345],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 4: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['ids_f'],
            ['ids_i'],
            name='',
            domain='',
            to=6,
        ),
        # 5: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['MASKS', 'ids_i'],
            ['packed'],
            name='',
            domain='',
            axis=0,
        ),
        # 6: Einsum inputs=3 outputs=1
        _node(
            'Einsum',
            ['packed', 'V', 'V'],
            ['output'],
            name='',
            domain='',
            equation='bc,r,w->bcrw',
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
            _tensor('LIKE', TensorProto.FLOAT16, (1, 1), INIT_LIKE_0),
            _tensor('SENT', TensorProto.INT32, (1, 1), INIT_SENT_1),
            _tensor('MASKS', TensorProto.INT64, (18,), INIT_MASKS_2),
            _tensor('V', TensorProto.INT64, (30,), INIT_V_3),
        ],
        value_info=[
            _vi('rng_f', TensorProto.FLOAT16, [1, 1]),
            _vi('case_i', TensorProto.INT32, [1, 1]),
            _vi('key_seq', TensorProto.INT32, [1, 6]),
            _vi('ids_f', TensorProto.FLOAT, [1, 10]),
            _vi('ids_i', TensorProto.INT32, [1, 10]),
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
