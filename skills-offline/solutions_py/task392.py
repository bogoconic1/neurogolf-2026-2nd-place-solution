from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task392'
TASK_NUM = 392
KAGGLE = {'score': 19.601837, 'date': '2026-07-14'}
MEMORY_BYTES = 180
PARAMS = 41
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task392_p0399_all_distance_safe221'
OPSETS = [('', 20)]


# score_period_i8: INT8[1, 1, 36, 1], 36 value(s)
INIT_SCORE_PERIOD_I8_0 = [-12, -12, 10, -9, -9, 4, -3, -3, 2, -1, -1, 2, -3, -3, 4, -7, -7, 10, -11, 8, -7, 6, -5, 4, -3, 2, -1, 2, -3, 4, -5, 6,
 -7, 8, -9, 12]

# axis_step2_i32: INT32[1], 1 value(s)
INIT_AXIS_STEP2_I32_1 = [2]

# bernoulli_p: FLOAT16[1, 1, 1, 1], 1 value(s)
INIT_BERNOULLI_P_2 = [0.875]

# zero_i8: INT8[1, 1, 1, 1], 1 value(s)
INIT_ZERO_I8_3 = [0]

# gray_i8: INT8[1, 1, 1, 1], 1 value(s)
INIT_GRAY_I8_4 = [-1]

# p0399: FLOAT16[1], 1 value(s)
INIT_P0399_5 = [0.3994140625]


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
            ['p0399'],
            ['h'],
            name='h_stream',
            domain='',
            dtype=2,
            seed=309147392.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0399'],
            ['axis'],
            name='axis_stream',
            domain='',
            dtype=9,
            seed=372124448.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0399'],
            ['d0'],
            name='d0_stream',
            domain='',
            dtype=2,
            seed=125052120.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0399'],
            ['d1'],
            name='d1_stream',
            domain='',
            dtype=2,
            seed=889410816.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0399'],
            ['d2'],
            name='d2_stream',
            domain='',
            dtype=2,
            seed=41459688.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0399'],
            ['d3'],
            name='d3_stream',
            domain='',
            dtype=2,
            seed=3412639744.0,
        ),
        # 6: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['h'],
            ['h17'],
            name='h17',
            domain='',
            bias=-16.0,
            lambd=0.5,
        ),
        # 7: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['h17'],
            ['B'],
            name='B',
            domain='',
            bias=9.0,
            lambd=-100.0,
        ),
        # 8: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['d2'],
            ['d2w'],
            name='d2w',
            domain='',
            bias=-2.0,
            lambd=0.5,
        ),
        # 9: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['d3'],
            ['d3w'],
            name='d3w',
            domain='',
            bias=-3.0,
            lambd=0.5,
        ),
        # 10: Add inputs=2 outputs=1
        _node(
            'Add',
            ['d0', 'd1'],
            ['d01'],
            name='d01',
            domain='',
        ),
        # 11: Add inputs=2 outputs=1
        _node(
            'Add',
            ['d01', 'd2w'],
            ['d012'],
            name='d012',
            domain='',
        ),
        # 12: Add inputs=2 outputs=1
        _node(
            'Add',
            ['d012', 'd3w'],
            ['dval'],
            name='dval',
            domain='',
        ),
        # 13: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['B', 'dval'],
            ['Bminusd'],
            name='Bminusd',
            domain='',
        ),
        # 14: Where inputs=3 outputs=1
        _node(
            'Where',
            ['axis', 'B', 'Bminusd'],
            ['start_r_i8'],
            name='start_r',
            domain='',
        ),
        # 15: Where inputs=3 outputs=1
        _node(
            'Where',
            ['axis', 'Bminusd', 'B'],
            ['start_c_i8'],
            name='start_c',
            domain='',
        ),
        # 16: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['start_r_i8'],
            ['start_r_i32'],
            name='',
            domain='',
            to=6,
        ),
        # 17: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['start_c_i8'],
            ['start_c_i32'],
            name='',
            domain='',
            to=6,
        ),
        # 18: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['start_r_i32'],
            ['end_r_i32'],
            name='end_r',
            domain='',
            bias=10.0,
            lambd=-100.0,
        ),
        # 19: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['start_c_i32'],
            ['end_c_i32'],
            name='end_c',
            domain='',
            bias=10.0,
            lambd=-100.0,
        ),
        # 20: Slice inputs=4 outputs=1
        _node(
            'Slice',
            ['score_period_i8', 'start_r_i32', 'end_r_i32', 'axis_step2_i32'],
            ['row_score_i8'],
            name='row_score',
            domain='',
        ),
        # 21: Slice inputs=4 outputs=1
        _node(
            'Slice',
            ['score_period_i8', 'start_c_i32', 'end_c_i32', 'axis_step2_i32'],
            ['col_pre_i8'],
            name='col_pre',
            domain='',
        ),
        # 22: Transpose inputs=1 outputs=1
        _node(
            'Transpose',
            ['col_pre_i8'],
            ['col_score_i8'],
            name='col_score',
            domain='',
            perm=[0, 1, 3, 2],
        ),
        # 23: Add inputs=2 outputs=1
        _node(
            'Add',
            ['row_score_i8', 'col_score_i8'],
            ['score_i8'],
            name='score',
            domain='',
        ),
        # 24: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['bernoulli_p'],
            ['color1_i8'],
            name='color1_bit',
            domain='',
            dtype=3,
            seed=11884942.0,
        ),
        # 25: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['bernoulli_p'],
            ['color2_i8'],
            name='color2_bit',
            domain='',
            dtype=3,
            seed=724427.0,
        ),
        # 26: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['bernoulli_p'],
            ['color3_i8'],
            name='color3_bit',
            domain='',
            dtype=3,
            seed=11185073.0,
        ),
        # 27: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['bernoulli_p'],
            ['color4_i8'],
            name='color4_bit',
            domain='',
            dtype=3,
            seed=1149.0,
        ),
        # 28: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['bernoulli_p'],
            ['color6_i8'],
            name='color6_bit',
            domain='',
            dtype=3,
            seed=392.0,
        ),
        # 29: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['bernoulli_p'],
            ['color7_i8'],
            name='color7_bit',
            domain='',
            dtype=3,
            seed=13981592.0,
        ),
        # 30: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['bernoulli_p'],
            ['color8_i8'],
            name='color8_bit',
            domain='',
            dtype=3,
            seed=1403410.0,
        ),
        # 31: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['bernoulli_p'],
            ['color9_i8'],
            name='color9_bit',
            domain='',
            dtype=3,
            seed=13318473.0,
        ),
        # 32: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['zero_i8', 'color1_i8', 'color2_i8', 'color3_i8', 'color4_i8', 'gray_i8', 'color6_i8', 'color7_i8',
             'color8_i8', 'color9_i8'],
            ['conv_kernel_i8'],
            name='assemble_color_kernel',
            domain='',
            axis=0,
        ),
        # 33: ConvInteger inputs=2 outputs=1
        _node(
            'ConvInteger',
            ['score_i8', 'conv_kernel_i8'],
            ['output'],
            name='output',
            domain='',
            pads=[0, 0, 20, 20],
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
            _tensor('score_period_i8', TensorProto.INT8, (1, 1, 36, 1), INIT_SCORE_PERIOD_I8_0),
            _tensor('axis_step2_i32', TensorProto.INT32, (1,), INIT_AXIS_STEP2_I32_1),
            _tensor('bernoulli_p', TensorProto.FLOAT16, (1, 1, 1, 1), INIT_BERNOULLI_P_2),
            _tensor('zero_i8', TensorProto.INT8, (1, 1, 1, 1), INIT_ZERO_I8_3),
            _tensor('gray_i8', TensorProto.INT8, (1, 1, 1, 1), INIT_GRAY_I8_4),
            _tensor('p0399', TensorProto.FLOAT16, (1,), INIT_P0399_5),
        ],
        value_info=[
            _vi('h', TensorProto.UINT8, [1]),
            _vi('axis', TensorProto.BOOL, [1]),
            _vi('d0', TensorProto.UINT8, [1]),
            _vi('d1', TensorProto.UINT8, [1]),
            _vi('d2', TensorProto.UINT8, [1]),
            _vi('d3', TensorProto.UINT8, [1]),
            _vi('h17', TensorProto.UINT8, [1]),
            _vi('B', TensorProto.UINT8, [1]),
            _vi('d2w', TensorProto.UINT8, [1]),
            _vi('d3w', TensorProto.UINT8, [1]),
            _vi('d01', TensorProto.UINT8, [1]),
            _vi('d012', TensorProto.UINT8, [1]),
            _vi('dval', TensorProto.UINT8, [1]),
            _vi('Bminusd', TensorProto.UINT8, [1]),
            _vi('start_r_i8', TensorProto.UINT8, [1]),
            _vi('start_c_i8', TensorProto.UINT8, [1]),
            _vi('start_r_i32', TensorProto.INT32, [1]),
            _vi('start_c_i32', TensorProto.INT32, [1]),
            _vi('end_r_i32', TensorProto.INT32, [1]),
            _vi('end_c_i32', TensorProto.INT32, [1]),
            _vi('row_score_i8', TensorProto.INT8, [1, 1, 10, 1]),
            _vi('col_pre_i8', TensorProto.INT8, [1, 1, 10, 1]),
            _vi('col_score_i8', TensorProto.INT8, [1, 1, 1, 10]),
            _vi('score_i8', TensorProto.INT8, [1, 1, 10, 10]),
            _vi('color1_i8', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('color2_i8', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('color3_i8', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('color4_i8', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('color6_i8', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('color7_i8', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('color8_i8', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('color9_i8', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('conv_kernel_i8', TensorProto.INT8, [10, 1, 1, 1]),
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
