from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task107'
TASK_NUM = 107
KAGGLE = {'score': 18.983843, 'date': '2026-07-12'}
MEMORY_BYTES = 284
PARAMS = 126
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task107_sneg_fold_v2_final_scaled'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task107_sneg_fold_v2_final_scaled'
OPSETS = [('', 18)]


# F2: FLOAT[10, 2], 20 value(s)
INIT_F2_0 = [0.9999989867210388, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0, 9.0]

# qsrc: FLOAT[1, 2, 6], 12 value(s)
INIT_QSRC_1 = [1.0, 1.0, 1.0, 1.0, 1.0, -15.358000755310059, 0.5, 1.5, 2.5, 3.5, 4.5, -2.13529372215271]

# qroi_num: FLOAT[2], 2 value(s)
INIT_QROI_NUM_2 = [0.0, 5.849999904632568]

# qsize: INT64[1], 1 value(s)
INIT_QSIZE_3 = [30]

# four8: INT8[1], 1 value(s)
INIT_FOUR8_4 = [8]

# sixteenf: FLOAT[1], 1 value(s)
INIT_SIXTEENF_5 = [8.0]

# X: FLOAT[2, 30], 60 value(s)
INIT_X_6 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5,
 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5]

# R1: FLOAT[2, 2], 4 value(s)
INIT_R1_7 = [1.0, 1.0, 1.0, 0.0]

# Bs1: FLOAT[2, 2], 4 value(s)
INIT_BS1_8 = [2.25, -0.75, -1.0, 0.5]

# Bs2: FLOAT[2, 2], 4 value(s)
INIT_BS2_9 = [2.0625, 0.375, -0.375, -0.25]

# fd_scale2: FLOAT[1], 1 value(s)
INIT_FD_SCALE2_10 = [1.3967386484146118]

# GA: FLOAT[2, 2, 2, 2], 16 value(s)
INIT_GA_11 = [-0.25, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.25, -0.0, 0.25, -0.0, -0.0, -0.0]


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
        # 0: Einsum inputs=16 outputs=1
        _node(
            'Einsum',
            ['qsrc', 'Bs1', 'X', 'X', 'input', 'input', 'X', 'X', 'Bs1', 'GA', 'sixteenf', 'GA', 'R1', 'GA',
             'sixteenf', 'GA'],
            ['fsw'],
            name='qsrcsum_scaled_weighted_single_moment_factor_score',
            domain='',
            equation='XYZ,ab,ah,bh,nchw,ncij,di,ei,de,PQPQ,O,uuOO,uv,vvOO,O,RSRS->n',
        ),
        # 1: QuantizeLinear inputs=3 outputs=1
        _node(
            'QuantizeLinear',
            ['fsw', 'sixteenf', 'four8'],
            ['fs2i'],
            name='',
            domain='',
        ),
        # 2: QuantizeLinear inputs=3 outputs=1
        _node(
            'QuantizeLinear',
            ['fsw', 'fd_scale2', 'fs2i'],
            ['f8'],
            name='',
            domain='',
        ),
        # 3: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['f8'],
            ['f32'],
            name='',
            domain='',
            to=1,
        ),
        # 4: Div inputs=2 outputs=1
        _node(
            'Div',
            ['qroi_num', 'f32'],
            ['qroi'],
            name='',
            domain='',
        ),
        # 5: Resize inputs=4 outputs=1
        _node(
            'Resize',
            ['qsrc', 'qroi', '', 'qsize'],
            ['H'],
            name='',
            domain='',
            axes=[2],
            coordinate_transformation_mode='tf_crop_and_resize',
            extrapolation_value=0.0,
            mode='nearest',
            nearest_mode='floor',
        ),
        # 6: Einsum inputs=38 outputs=1
        _node(
            'Einsum',
            ['GA', 'GA', 'GA', 'sixteenf', 'sixteenf', 'sixteenf', 'sixteenf', 'GA', 'R1', 'R1', 'F2', 'input',
             'X', 'X', 'GA', 'R1', 'GA', 'GA', 'R1', 'R1', 'GA', 'R1', 'Bs1', 'R1', 'qsrc', 'qsrc', 'qsrc',
             'qsrc', 'qsrc', 'Bs2', 'Bs1', 'R1', 'Bs1', 'R1', 'R1', 'Bs2', 'GA', 'f32'],
            ['orientf32'],
            name='factor_scaled_homogeneous_occupancy_orientation',
            domain='',
            equation='XYXY,CICI,ZVZV,O,O,O,O,AAOO,AP,PU,cU,nchw,Eh,Dw,TDEG,BT,BBOO,LLOO,LM,MG,mmOO,lm,kl,vk,QRS,QRS,QRS,QRS,QRS,op,pq,qr,rs,st,tu,uv,FJFJ,z->n',
        ),
        # 7: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['orientf32'],
            ['d8'],
            name='',
            domain='',
            to=3,
        ),
        # 8: Abs inputs=1 outputs=1
        _node(
            'Abs',
            ['d8'],
            ['ad8'],
            name='',
            domain='',
        ),
        # 9: Concat inputs=2 outputs=1
        _node(
            'Concat',
            ['sixteenf', 'orientf32'],
            ['D'],
            name='',
            domain='',
            axis=0,
        ),
        # 10: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['f8', 'four8'],
            ['fourf8'],
            name='',
            domain='',
        ),
        # 11: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['fourf8', 'ad8'],
            ['e8'],
            name='',
            domain='',
        ),
        # 12: Concat inputs=2 outputs=1
        _node(
            'Concat',
            ['four8', 'e8'],
            ['E8'],
            name='',
            domain='',
            axis=0,
        ),
        # 13: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['E8'],
            ['Eoff'],
            name='',
            domain='',
            to=1,
        ),
        # 14: Einsum inputs=76 outputs=1
        _node(
            'Einsum',
            ['R1', 'GA', 'X', 'X', 'Bs1', 'H', 'H', 'X', 'Bs2', 'H', 'X', 'R1', 'GA', 'H', 'X', 'Bs1', 'H', 'X',
             'Bs2', 'H', 'GA', 'input', 'X', 'R1', 'X', 'Bs1', 'H', 'H', 'X', 'Bs2', 'H', 'X', 'Bs1', 'H', 'X',
             'Bs2', 'H', 'X', 'R1', 'GA', 'H', 'F2', 'F2', 'F2', 'GA', 'R1', 'F2', 'GA', 'R1', 'GA', 'F2', 'R1',
             'Bs1', 'Bs1', 'GA', 'GA', 'Bs2', 'F2', 'X', 'GA', 'X', 'D', 'X', 'GA', 'X', 'D', 'GA', 'X', 'GA',
             'X', 'Eoff', 'GA', 'X', 'GA', 'X', 'Eoff'],
            ['output'],
            name='v13_qlinear_bias_fold_direct_renderer',
            domain='',
            equation='fg,ffnn,fi,ki,kl,...gr,...lr,mi,mo,...or,di,de,eenn,...er,qi,pq,...pr,bi,ab,...ar,vvnn,...Yij,vj,vw,xj,xy,...ws,...ys,zj,zA,...As,Cj,BC,...Bs,Hj,FH,...Fs,tj,tu,uunn,...us,YV,YW,YN,NNnn,UV,cU,VVnn,WX,XXnn,cX,WZ,ZZ,ZZ,QQnn,QNhJ,JJ,cJ,Dr,hDEG,Es,G,Ir,hIKL,Ks,L,MMnn,Mr,hMOP,Os,P,RRnn,Rr,hRST,Ss,T->...crs',
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
            _tensor('F2', TensorProto.FLOAT, (10, 2), INIT_F2_0),
            _tensor('qsrc', TensorProto.FLOAT, (1, 2, 6), INIT_QSRC_1),
            _tensor('qroi_num', TensorProto.FLOAT, (2,), INIT_QROI_NUM_2),
            _tensor('qsize', TensorProto.INT64, (1,), INIT_QSIZE_3),
            _tensor('four8', TensorProto.INT8, (1,), INIT_FOUR8_4),
            _tensor('sixteenf', TensorProto.FLOAT, (1,), INIT_SIXTEENF_5),
            _tensor('X', TensorProto.FLOAT, (2, 30), INIT_X_6),
            _tensor('R1', TensorProto.FLOAT, (2, 2), INIT_R1_7),
            _tensor('Bs1', TensorProto.FLOAT, (2, 2), INIT_BS1_8),
            _tensor('Bs2', TensorProto.FLOAT, (2, 2), INIT_BS2_9),
            _tensor('fd_scale2', TensorProto.FLOAT, (1,), INIT_FD_SCALE2_10),
            _tensor('GA', TensorProto.FLOAT, (2, 2, 2, 2), INIT_GA_11),
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
