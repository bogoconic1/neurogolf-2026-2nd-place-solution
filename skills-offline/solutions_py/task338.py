from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task338'
TASK_NUM = 338
KAGGLE = {'score': 19.900134, 'date': '2026-07-14'}
MEMORY_BYTES = 0
PARAMS = 164
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task338_moment_1_2'
OPSETS = [('', 18)]


# C: FLOAT[10, 2], 20 value(s)
INIT_C_0 = [-1.0, 0.009011914022266865, 0.0, 0.0, -0.05077160522341728, 0.05077160522341728, 1.0, -0.009011914022266865, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# B: FLOAT[30, 4], 120 value(s)
INIT_B_1 = [-0.26764944195747375, 0.08883871883153915, 0.1044023409485817, -0.35096633434295654, -0.8013447523117065,
 0.5143234133720398, 0.348345011472702, 0.5621986985206604, 0.6646956205368042, 0.06241312250494957,
 -0.3072149157524109, -0.3218962252140045, -0.2781456410884857, 0.25316670536994934, 0.05917270481586456,
 0.10545303672552109, 0.26046547293663025, -0.21586035192012787, -0.04376714304089546, -0.24619069695472717,
 0.0219033844769001, -0.08894763141870499, -0.14501537382602692, -0.11579670011997223, 0.5731824040412903,
 -0.3652632236480713, -0.13730241358280182, 0.2946647107601166, -0.5818364024162292, 0.40521731972694397,
 0.030266981571912766, 0.008774589747190475, 0.4948371350765228, -0.4794341027736664, -0.475931853055954,
 -0.17096902430057526, -0.5968469381332397, 0.392614483833313, 0.15519115328788757, -0.014295682311058044,
 0.3508482873439789, -0.30194199085235596, 0.1079859733581543, 0.21632979810237885, -0.07543810456991196,
 0.2015056312084198, -0.4142906069755554, -0.11986607313156128, -0.1187458485364914, -0.30646389722824097,
 0.03466354310512543, -0.41540905833244324, 0.7020540237426758, 0.5395157933235168, 0.02225973643362522,
 0.3687618374824524, -0.06979310512542725, -0.14373233914375305, 0.127354234457016, 0.5812699198722839,
 0.3303382098674774, 0.7560962438583374, 0.5258561968803406, -0.5722780823707581, -0.12567564845085144,
 -0.7560616731643677, -0.44310179352760315, 0.3164062201976776, -0.38183659315109253, 0.10316766798496246,
 0.09675905853509903, -0.40308916568756104, -1.3184040784835815, 0.6589854955673218, 1.4323080778121948,
 0.7384654879570007, 2.573298931121826, 0.7069480419158936, -1.6381127834320068, -1.2259435653686523,
 -2.6202619075775146, -0.8421676158905029, 1.511960506439209, 0.1271754503250122, 1.2716615200042725,
 -0.6507390141487122, -1.2162015438079834, 0.7425261735916138, -1.1241004467010498, 0.8140149116516113,
 1.134700059890747, -1.922951340675354, 1.4678951501846313, -1.0576022863388062, -1.6857306957244873,
 1.8604463338851929, -1.6529829502105713, 1.2452882528305054, 1.5512194633483887, -1.09478759765625, -2.759036064147949,
 0.30686426162719727, 1.0105074644088745, -1.0207599401474, -0.34231624007225037, -0.17881177365779877,
 0.7915186285972595, -0.9291219711303711, 1.6184203624725342, -1.2337679862976074, -1.0765867233276367,
 1.2240625619888306, 2.9281487464904785, -0.18053823709487915, -0.7979715466499329, 0.7110311388969421,
 -1.5975507497787476, 0.076698437333107, -0.4904351532459259, 1.606986165046692]

# V: FLOAT[4, 4], 16 value(s)
INIT_V_2 = [28958.30859375, -15511.150390625, 18557.388671875, 5825.71533203125, -185565.40625, 26006.595703125, 821.4503173828125,
 -4698.1533203125, 85850.921875, 27076.775390625, 18293.74609375, -16865.521484375, 20387.998046875, -58503.1171875,
 -4088.99609375, -3614.80859375]

# W: FLOAT[2, 4], 8 value(s)
INIT_W_3 = [2.673267545105773e-06, 3.2955094866338186e-06, 4.002259447588585e-05, 1.2449430869310163e-05, -1.0, 1.0, 1.0, 1.0]


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
        # 0: Einsum inputs=15 outputs=1
        _node(
            'Einsum',
            ['input', 'C', 'B', 'input', 'C', 'B', 'input', 'C', 'B', 'B', 'B', 'B', 'V', 'W', 'C'],
            ['output'],
            name='output',
            domain='',
            equation='...arl,aq,li,...bru,bq,ui,...cvs,cq,vj,xi,xk,xk,jk,qk,ow->...ors',
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
            _tensor('C', TensorProto.FLOAT, (10, 2), INIT_C_0),
            _tensor('B', TensorProto.FLOAT, (30, 4), INIT_B_1),
            _tensor('V', TensorProto.FLOAT, (4, 4), INIT_V_2),
            _tensor('W', TensorProto.FLOAT, (2, 4), INIT_W_3),
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
