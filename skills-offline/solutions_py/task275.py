from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task275'
TASK_NUM = 275
KAGGLE = {'score': 19.561921, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 230
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'm'
OPSETS = [('', 12)]


# B: FLOAT[2, 3, 30], 180 value(s)
INIT_B_0 = [3.298630475997925, 3.4987361431121826, 3.6988418102264404, 2.464031457901001, 2.6641368865966797, 2.8642425537109375,
 1.6294323205947876, 1.8295378684997559, 2.0296432971954346, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.298630475997925, -4.832069396972656, -7.36550760269165,
 -11.464031219482422, -13.997469902038574, -16.530908584594727, -20.629432678222656, -23.162870407104492,
 -25.69631004333496, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 1.983699943446042e-16, 4.3333330154418945, 8.666666030883789, 1.4817965262509168e-16, 4.3333330154418945,
 8.666666030883789, 9.798929105324445e-17, 4.3333330154418945, 8.666666030883789, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.298630475997925, 3.4987361431121826,
 3.6988418102264404, 3.898947238922119, 2.464031457901001, 2.6641368865966797, 2.8642425537109375, 3.0643482208251953,
 1.6294323205947876, 1.8295378684997559, 2.0296432971954346, 2.2297489643096924, 0.7948331236839294, 0.9949386715888977,
 1.1950442790985107, 1.395149827003479, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 -2.298630475997925, -4.832069396972656, -7.36550760269165, -9.898946762084961, -11.464031219482422,
 -13.997469902038574, -16.530908584594727, -19.064346313476562, -20.629432678222656, -23.162870407104492,
 -25.69631004333496, -28.229747772216797, -29.794832229614258, -32.328269958496094, -34.86170959472656,
 -37.39514923095703, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.983699943446042e-16,
 4.3333330154418945, 8.666666030883789, 12.999999046325684, 1.4817965262509168e-16, 4.3333330154418945,
 8.666666030883789, 12.999999046325684, 9.798929105324445e-17, 4.3333330154418945, 8.666666030883789,
 12.999999046325684, 4.7798939407564574e-17, 4.3333330154418945, 8.666666030883789, 12.999999046325684, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# S: FLOAT[2, 3, 2], 12 value(s)
INIT_S_1 = [-0.614061713218689, -0.06842171400785446, -0.048502128571271896, -0.09818800538778305, -1.5265566588595902e-16,
 -0.05424501746892929, 0.5456400513648987, 1.3281558503828072e-17, -0.049685847014188766, 0.0, -0.2750997543334961,
 -0.2208547443151474]

# E: FLOAT[3, 10], 30 value(s)
INIT_E_2 = [12.87308406829834, -10.644346237182617, -10.59564208984375, -11.791507720947266, -11.540828704833984,
 -1.070996297998492e-23, 1.3814482947309873e-22, -4.714196418890254e-22, 12.989385604858398, -2.778744552612375e-23,
 2.6345596313476562, -62.637821197509766, 60.28398132324219, -17.185260772705078, 28.698076248168945,
 3.829777169770544e-21, -2.8310739121283623e-21, 3.8157156648126396e-20, 7.628183841705322, -8.994630916395386e-22,
 8.419803619384766, -1.0697612762451172, -0.2958490550518036, -3.525651693344116, -2.782616138458252,
 1.7731304670193186e-20, 4.518851739563461e-22, 5.693879369205118e-22, -37.79075622558594, 1.9567973190352305e-21]

# L0s: FLOAT[2, 2, 2], 8 value(s)
INIT_L0S_3 = [0.04304530844092369, 0.043045300990343094, -0.0, -0.25827184319496155, 0.3874078094959259, -0.12913593649864197,
 -0.08609061688184738, 6.449828049426287e-08]


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
        # 0: Einsum inputs=46 outputs=1
        _node(
            'Einsum',
            ['input', 'B', 'B', 'input', 'E', 'B', 'S', 'L0s', 'B', 'S', 'B', 'S', 'S', 'B', 'S', 'L0s', 'B',
             'S', 'B', 'S', 'S', 'L0s', 'S', 'S', 'S', 'E', 'E', 'S', 'E', 'S', 'L0s', 'S', 'B', 'S', 'B', 'S',
             'S', 'B', 'L0s', 'S', 'B', 'S', 'B', 'S', 'S', 'B'],
            ['output'],
            name='output',
            domain='',
            equation='nzHW,mRH,mCW,nchw,tc,mAh,gAa,Mxa,mBh,gBM,mDh,gDd,dkM,mJw,gJj,Nyj,mKw,gKN,mLw,gLl,luN,gbf,bGb,PtP,ptp,to,Zo,OZO,Yo,TYT,Mxe,pEe,mEr,pFM,mFr,iSM,pIi,mIr,Nyq,pQq,mQs,pUN,mUs,vXN,pVv,mVs->nors',
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
            _tensor('B', TensorProto.FLOAT, (2, 3, 30), INIT_B_0),
            _tensor('S', TensorProto.FLOAT, (2, 3, 2), INIT_S_1),
            _tensor('E', TensorProto.FLOAT, (3, 10), INIT_E_2),
            _tensor('L0s', TensorProto.FLOAT, (2, 2, 2), INIT_L0S_3),
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
