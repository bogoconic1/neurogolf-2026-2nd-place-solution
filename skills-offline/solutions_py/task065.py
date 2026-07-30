from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task065'
TASK_NUM = 65
KAGGLE = {'score': 20.163718, 'date': '2026-07-15'}
MEMORY_BYTES = 121
PARAMS = 5
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task065_seeded_rng_cycle'
OPSETS = [('', 21)]


# p0: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P0_0 = [0.2452748566865921]

# p1: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P1_1 = [0.602257490158081]

# s32: FLOAT[1], 1 value(s)
INIT_S32_2 = [32.0]

# zm1: INT8[1], 1 value(s)
INIT_ZM1_3 = [-1]

# zm32: INT8[1], 1 value(s)
INIT_ZM32_4 = [-32]


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
        # 0: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0'],
            ['U1'],
            name='n_U1',
            domain='',
            dtype=3,
            seed=213637.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0'],
            ['U2'],
            name='n_U2',
            domain='',
            dtype=3,
            seed=1189965184.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0'],
            ['U3'],
            name='n_U3',
            domain='',
            dtype=3,
            seed=2101366528.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['U6'],
            name='n_U6',
            domain='',
            dtype=3,
            seed=11050361.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['R0'],
            name='n_R0',
            domain='',
            dtype=3,
            seed=773202176.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['R1'],
            name='n_R1',
            domain='',
            dtype=3,
            seed=3469414912.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['R2'],
            name='n_R2',
            domain='',
            dtype=3,
            seed=280529600.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['R3'],
            name='n_R3',
            domain='',
            dtype=3,
            seed=70477400.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['R4'],
            name='n_R4',
            domain='',
            dtype=3,
            seed=83347776.0,
        ),
        # 9: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['R6'],
            name='n_R6',
            domain='',
            dtype=3,
            seed=452893.0,
        ),
        # 10: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['C0'],
            name='n_C0',
            domain='',
            dtype=3,
            seed=881204352.0,
        ),
        # 11: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['C1'],
            name='n_C1',
            domain='',
            dtype=3,
            seed=2201297664.0,
        ),
        # 12: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['C2'],
            name='n_C2',
            domain='',
            dtype=3,
            seed=2710272768.0,
        ),
        # 13: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['C3'],
            name='n_C3',
            domain='',
            dtype=3,
            seed=3727982848.0,
        ),
        # 14: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['C4'],
            name='n_C4',
            domain='',
            dtype=3,
            seed=34004700.0,
        ),
        # 15: QLinearMatMul inputs=8 outputs=1
        _node(
            'QLinearMatMul',
            ['U1', 's32', 'zm1', 'R0', 's32', 'zm1', 's32', 'zm32'],
            ['row0'],
            name='row0',
            domain='',
        ),
        # 16: QLinearMatMul inputs=8 outputs=1
        _node(
            'QLinearMatMul',
            ['U1', 's32', 'zm1', 'R1', 's32', 'zm1', 's32', 'zm32'],
            ['row1'],
            name='row1',
            domain='',
        ),
        # 17: QLinearMatMul inputs=8 outputs=1
        _node(
            'QLinearMatMul',
            ['U2', 's32', 'zm1', 'R2', 's32', 'zm1', 's32', 'zm32'],
            ['row2'],
            name='row2',
            domain='',
        ),
        # 18: QLinearMatMul inputs=8 outputs=1
        _node(
            'QLinearMatMul',
            ['U3', 's32', 'zm1', 'R3', 's32', 'zm1', 's32', 'zm32'],
            ['row3'],
            name='row3',
            domain='',
        ),
        # 19: QLinearMatMul inputs=8 outputs=1
        _node(
            'QLinearMatMul',
            ['U3', 's32', 'zm1', 'R4', 's32', 'zm1', 's32', 'zm32'],
            ['row4'],
            name='row4',
            domain='',
        ),
        # 20: QLinearMatMul inputs=8 outputs=1
        _node(
            'QLinearMatMul',
            ['U6', 's32', 'zm1', 'R6', 's32', 'zm1', 's32', 'zm32'],
            ['row6'],
            name='row6',
            domain='',
        ),
        # 21: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['B5a'],
            name='n_B5a',
            domain='',
            dtype=3,
            seed=8321429.0,
        ),
        # 22: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['B5b'],
            name='n_B5b',
            domain='',
            dtype=3,
            seed=8880904.0,
        ),
        # 23: Add inputs=2 outputs=1
        _node(
            'Add',
            ['B5a', 'B5b'],
            ['col5'],
            name='n_col5',
            domain='',
        ),
        # 24: Concat inputs=7 outputs=1
        _node(
            'Concat',
            ['row0', 'row1', 'row2', 'row3', 'row4', 'col5', 'row6'],
            ['row'],
            name='',
            domain='',
            axis=2,
        ),
        # 25: Add inputs=2 outputs=1
        _node(
            'Add',
            ['U1', 'C0'],
            ['col0'],
            name='',
            domain='',
        ),
        # 26: Add inputs=2 outputs=1
        _node(
            'Add',
            ['U1', 'C1'],
            ['col1'],
            name='',
            domain='',
        ),
        # 27: Add inputs=2 outputs=1
        _node(
            'Add',
            ['U2', 'C2'],
            ['col2'],
            name='',
            domain='',
        ),
        # 28: Add inputs=2 outputs=1
        _node(
            'Add',
            ['U3', 'C3'],
            ['col3'],
            name='',
            domain='',
        ),
        # 29: Add inputs=2 outputs=1
        _node(
            'Add',
            ['U3', 'C4'],
            ['col4'],
            name='',
            domain='',
        ),
        # 30: Concat inputs=7 outputs=1
        _node(
            'Concat',
            ['col0', 'col1', 'col2', 'col3', 'col4', 'col5', 'U6'],
            ['col'],
            name='',
            domain='',
            axis=3,
        ),
        # 31: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['row', 'col'],
            ['X'],
            name='',
            domain='',
        ),
        # 32: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['V0'],
            name='n_V0',
            domain='',
            dtype=3,
            seed=-114242344.0,
        ),
        # 33: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['V1'],
            name='n_V1',
            domain='',
            dtype=3,
            seed=-15931023.0,
        ),
        # 34: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['V2'],
            name='n_V2',
            domain='',
            dtype=3,
            seed=754972224.0,
        ),
        # 35: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['V3'],
            name='n_V3',
            domain='',
            dtype=3,
            seed=19182136.0,
        ),
        # 36: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['V4'],
            name='n_V4',
            domain='',
            dtype=3,
            seed=9285648.0,
        ),
        # 37: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['V5'],
            name='n_V5',
            domain='',
            dtype=3,
            seed=-6669849.0,
        ),
        # 38: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['V6'],
            name='n_V6',
            domain='',
            dtype=3,
            seed=125294696.0,
        ),
        # 39: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['V7'],
            name='n_V7',
            domain='',
            dtype=3,
            seed=125037576.0,
        ),
        # 40: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['V8'],
            name='n_V8',
            domain='',
            dtype=3,
            seed=-19052922.0,
        ),
        # 41: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['w0'],
            name='n_w0',
            domain='',
            dtype=3,
            seed=7592611.0,
        ),
        # 42: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['w3'],
            name='n_w3',
            domain='',
            dtype=3,
            seed=8831200.0,
        ),
        # 43: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['V7', 'V1'],
            ['w1'],
            name='n_w1',
            domain='',
        ),
        # 44: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['V2', 'V6'],
            ['w2'],
            name='n_w2',
            domain='',
        ),
        # 45: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['V3', 'V2'],
            ['w4'],
            name='n_w4',
            domain='',
        ),
        # 46: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['V0', 'V4'],
            ['w5'],
            name='n_w5',
            domain='',
        ),
        # 47: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['V5', 'V0'],
            ['w6'],
            name='n_w6',
            domain='',
        ),
        # 48: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['V6', 'V5'],
            ['w7'],
            name='n_w7',
            domain='',
        ),
        # 49: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['V0', 'V7'],
            ['w8'],
            name='n_w8',
            domain='',
        ),
        # 50: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['V8', 'V0'],
            ['w9'],
            name='n_w9',
            domain='',
        ),
        # 51: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9'],
            ['W'],
            name='n_W',
            domain='',
            axis=0,
        ),
        # 52: ConvInteger inputs=2 outputs=1
        _node(
            'ConvInteger',
            ['X', 'W'],
            ['output'],
            name='',
            domain='',
            pads=[0, 0, 23, 23],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.INT32, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('p0', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P0_0),
            _tensor('p1', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P1_1),
            _tensor('s32', TensorProto.FLOAT, (1,), INIT_S32_2),
            _tensor('zm1', TensorProto.INT8, (1,), INIT_ZM1_3),
            _tensor('zm32', TensorProto.INT8, (1,), INIT_ZM32_4),
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
