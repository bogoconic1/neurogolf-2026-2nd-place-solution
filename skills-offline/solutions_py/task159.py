from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task159'
TASK_NUM = 159
KAGGLE = {'score': 19.340518, 'date': '2026-07-14'}
MEMORY_BYTES = 187
PARAMS = 100
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'orbit_shared_p103_28'
OPSETS = [('', 18)]


# one_h: FLOAT16[1, 1], 1 value(s)
INIT_ONE_H_0 = [0.5]

# axis_size: INT64[1], 1 value(s)
INIT_AXIS_SIZE_1 = [30]

# ch: FLOAT16[10, 2], 20 value(s)
INIT_CH_2 = [0.264892578125, -0.734375, 0.5048828125, 0.458984375, 0.6025390625, 1.310546875, 0.202392578125, 0.830078125,
 0.168212890625, 0.91552734375, 0.1243896484375, 0.98876953125, 0.07366943359375, 0.890625, 0.05096435546875,
 0.92626953125, 0.03460693359375, 1.03125, 0.01557159423828125, 1.0595703125]

# src_map: FLOAT16[2, 4, 3], 24 value(s)
INIT_SRC_MAP_3 = [0.70703125, 0.70703125, 0.70703125, 0.70703125, 0.70703125, 0.70703125, 0.70703125, 0.70703125, 0.70703125, 0.70703125,
 0.70703125, 0.70703125, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

# term_coef: FLOAT16[3, 2, 2], 12 value(s)
INIT_TERM_COEF_4 = [0.368896484375, 0.023162841796875, -0.1983642578125, -0.0111236572265625, 0.0160675048828125, 0.047393798828125,
 0.126953125, -0.00612640380859375, 0.0003876686096191406, 0.1280517578125, -0.302490234375, 0.0145721435546875]

# pack_code: FLOAT[1, 10], 10 value(s)
INIT_PACK_CODE_5 = [0.0, 8194.0, 0.07999999821186066, 8198.0, 8200.0, 8202.0, 8204.0, 8206.0, 8208.0, 8210.0]

# pack_base: FLOAT[1], 1 value(s)
INIT_PACK_BASE_6 = [4096.0]

# adapter: FLOAT16[3, 3], 9 value(s)
INIT_ADAPTER_7 = [-88.125, 816.0, 395.25, 137.375, -2334.0, 453.0, 32.40625, -859.5, -846.0]

# roi_slope: FLOAT16[1, 2], 2 value(s)
INIT_ROI_SLOPE_8 = [-0.25, 7.0]

# q1: FLOAT[1, 1], 1 value(s)
INIT_Q1_9 = [0.4227854907512665]

# q2: FLOAT[1, 1], 1 value(s)
INIT_Q2_10 = [0.5518640279769897]

# axis_base: FLOAT16[1, 2, 5], 10 value(s)
INIT_AXIS_BASE_11 = [0.98486328125, 1.0, 1.0, 0.02911376953125, 0.98486328125, 1.0, 0.51123046875, -0.0085601806640625, 1.0, 1.0]

# cat_code: FLOAT16[4, 2], 8 value(s)
INIT_CAT_CODE_12 = [-0.8505859375, 1.6640625, 0.042999267578125, 5.0234375, 2.892578125, -0.084228515625, -3.087890625, 3.041015625]


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
        # 0: Einsum inputs=2 outputs=1
        _node(
            'Einsum',
            ['input', 'pack_code'],
            ['pack_sum'],
            name='',
            domain='',
            equation='nchw,zc->nz',
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['pack_sum'],
            ['sprite_sum_u8'],
            name='',
            domain='',
            to=2,
        ),
        # 2: QuantizeLinear inputs=2 outputs=1
        _node(
            'QuantizeLinear',
            ['pack_sum', 'pack_base'],
            ['sprite_count_u8'],
            name='',
            domain='',
        ),
        # 3: Mod inputs=2 outputs=1
        _node(
            'Mod',
            ['sprite_sum_u8', 'sprite_count_u8'],
            ['m_idx_u8'],
            name='',
            domain='',
        ),
        # 4: Div inputs=2 outputs=1
        _node(
            'Div',
            ['sprite_sum_u8', 'sprite_count_u8'],
            ['color_u8'],
            name='',
            domain='',
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q1'],
            ['pat0'],
            name='b0',
            domain='',
            dtype=9,
            seed=56079200.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q1'],
            ['pat1'],
            name='b1',
            domain='',
            dtype=9,
            seed=17155020.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q1'],
            ['pat2'],
            name='b2',
            domain='',
            dtype=9,
            seed=11819663.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q1'],
            ['pat3'],
            name='b3',
            domain='',
            dtype=9,
            seed=-187693472.0,
        ),
        # 9: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['one_h'],
            ['pat4'],
            name='b4',
            domain='',
            dtype=9,
            seed=-94929584.0,
        ),
        # 10: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q2'],
            ['pat5'],
            name='b5x',
            domain='',
            dtype=9,
            seed=-373460960.0,
        ),
        # 11: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q1'],
            ['pat6'],
            name='b6',
            domain='',
            dtype=9,
            seed=80450064.0,
        ),
        # 12: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q2'],
            ['pat7'],
            name='b7',
            domain='',
            dtype=9,
            seed=-11094935.0,
        ),
        # 13: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['one_h'],
            ['pat8'],
            name='b8',
            domain='',
            dtype=9,
            seed=13642501.0,
        ),
        # 14: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['pat0', 'pat1', 'pat2'],
            ['row0'],
            name='row0',
            domain='',
            axis=1,
        ),
        # 15: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['pat3', 'pat4', 'pat5'],
            ['row1'],
            name='row1',
            domain='',
            axis=1,
        ),
        # 16: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['pat6', 'pat7', 'pat8'],
            ['row2'],
            name='row2',
            domain='',
            axis=1,
        ),
        # 17: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['row0', 'row1', 'row2'],
            ['crop_b'],
            name='crop_b',
            domain='',
            axis=0,
        ),
        # 18: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['crop_b'],
            ['crop_h'],
            name='crop_h',
            domain='',
            to=10,
        ),
        # 19: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['m_idx_u8'],
            ['m_h'],
            name='',
            domain='',
            to=10,
        ),
        # 20: Reciprocal inputs=1 outputs=1
        _node(
            'Reciprocal',
            ['m_h'],
            ['m_recip'],
            name='',
            domain='',
        ),
        # 21: Gemm inputs=3 outputs=1
        _node(
            'Gemm',
            ['m_recip', 'roi_slope', 'one_h'],
            ['axis_roi'],
            name='',
            domain='',
            beta=0.5,
        ),
        # 22: Resize inputs=4 outputs=1
        _node(
            'Resize',
            ['axis_base', 'axis_roi', '', 'axis_size'],
            ['axis_features'],
            name='',
            domain='',
            axes=[2],
            coordinate_transformation_mode='tf_crop_and_resize',
            extrapolation_value=0.0,
            mode='nearest',
            nearest_mode='floor',
        ),
        # 23: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['color_u8'],
            ['color_h'],
            name='',
            domain='',
            to=10,
        ),
        # 24: Concat inputs=2 outputs=1
        _node(
            'Concat',
            ['one_h', 'color_h'],
            ['color_aug'],
            name='',
            domain='',
            axis=1,
        ),
        # 25: Einsum inputs=33 outputs=1
        _node(
            'Einsum',
            ['crop_h', 'src_map', 'src_map', 'src_map', 'src_map', 'adapter', 'term_coef', 'term_coef',
             'color_aug', 'term_coef', 'color_aug', 'axis_features', 'axis_features', 'term_coef', 'cat_code',
             'axis_features', 'term_coef', 'term_coef', 'cat_code', 'axis_features', 'cat_code',
             'axis_features', 'axis_features', 'term_coef', 'cat_code', 'axis_features', 'term_coef',
             'term_coef', 'cat_code', 'axis_features', 'cat_code', 'ch', 'ch'],
            ['output'],
            name='',
            domain='',
            equation='ab,eta,eub,Gtc,Gud,fg,geG,fAC,nC,fBD,nD,nir,njr,hjI,tI,nlr,mlJ,oJK,tK,nwr,tw,nxs,nys,pyL,uL,nzs,qzM,vMN,uN,nWs,uW,kA,kB->nkrs',
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.FLOAT16, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('one_h', TensorProto.FLOAT16, (1, 1), INIT_ONE_H_0),
            _tensor('axis_size', TensorProto.INT64, (1,), INIT_AXIS_SIZE_1),
            _tensor('ch', TensorProto.FLOAT16, (10, 2), INIT_CH_2),
            _tensor('src_map', TensorProto.FLOAT16, (2, 4, 3), INIT_SRC_MAP_3),
            _tensor('term_coef', TensorProto.FLOAT16, (3, 2, 2), INIT_TERM_COEF_4),
            _tensor('pack_code', TensorProto.FLOAT, (1, 10), INIT_PACK_CODE_5),
            _tensor('pack_base', TensorProto.FLOAT, (1,), INIT_PACK_BASE_6),
            _tensor('adapter', TensorProto.FLOAT16, (3, 3), INIT_ADAPTER_7),
            _tensor('roi_slope', TensorProto.FLOAT16, (1, 2), INIT_ROI_SLOPE_8),
            _tensor('q1', TensorProto.FLOAT, (1, 1), INIT_Q1_9),
            _tensor('q2', TensorProto.FLOAT, (1, 1), INIT_Q2_10),
            _tensor('axis_base', TensorProto.FLOAT16, (1, 2, 5), INIT_AXIS_BASE_11),
            _tensor('cat_code', TensorProto.FLOAT16, (4, 2), INIT_CAT_CODE_12),
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
