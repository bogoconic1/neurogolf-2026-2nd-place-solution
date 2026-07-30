from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task343'
TASK_NUM = 343
KAGGLE = {'score': 20.072746, 'date': '2026-07-15'}
MEMORY_BYTES = 8
PARAMS = 130
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task343_direct_bernoulli_affine'
OPSETS = [('', 18)]


# p: FLOAT[2], 2 value(s)
INIT_P_0 = [-0.5106491446495056, 0.25532472133636475]

# U: FLOAT[2, 30, 2], 120 value(s)
INIT_U_1 = [-0.8173027634620667, 0.5497647523880005, -1.1439881324768066, 0.23123043775558472, -1.1431174278259277,
 -0.22371047735214233, -1.4935455322265625, -0.9912935495376587, 1.2423650026321411, 1.8469502925872803,
 0.1951480656862259, 0.965474545955658, 0.189178004860878, -0.9666633009910583, 0.5447033643722534, -0.8206846117973328,
 -0.8829817175865173, 0.5939441919326782, -0.9571214318275452, 0.19345970451831818, -0.816403329372406,
 -0.15977181494235992, 0.4487806558609009, 0.2978639006614685, -0.38128575682640076, -0.5668349862098694,
 -0.37050390243530273, -1.83302903175354, -0.4248107075691223, 2.170701265335083, 0.1814824491739273, 16.29501724243164,
 1.5032230615615845, 2.5030224323272705, 1.746754765510559, -2.648529529571533, 1.4045839309692383, -2.253946542739868,
 -0.39483222365379333, -15.436820983886719, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
 -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.2644382417201996, -0.9488400816917419, 0.7034302353858948,
 -0.6895004510879517, 0.9539386630058289, -0.24540981650352478, 0.9488400816917419, 0.2644382119178772,
 -0.6895004510879517, -0.7034302353858948, 0.24540981650352478, 0.9539386630058289, 7.933145980132394e-07,
 -2.846520146704279e-06, 2.1102907794556813e-06, -2.0685015442722943e-06, 0.8829817175865173, -0.22715546190738678,
 0.9571214318275452, 0.2667462229728699, 0.816403329372406, 0.832896888256073, -0.4487806558609009, -1.7444665431976318,
 0.38128575682640076, -1.3681045770645142, 0.37050390243530273, -0.3631669580936432, 0.4248107075691223,
 -0.1092866063117981, -2.7784383296966553, -2.3545501232147217, 0.16957443952560425, 9.590386390686035,
 -1.7579751014709473, -0.159598246216774, -0.11019422858953476, -1.8587923049926758, -0.12747858464717865,
 -1.8326008319854736, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
 -0.0, -0.0, -0.0, -0.0]

# C: FLOAT[2, 2, 2], 8 value(s)
INIT_C_2 = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0]


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
            ['p'],
            ['hmode'],
            name='direct_mode_bits',
            domain='',
            dtype=1,
            seed=104443568.0,
        ),
        # 1: Einsum inputs=71 outputs=1
        _node(
            'Einsum',
            ['C', 'hmode', 'hmode', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'C', 'C', 'U', 'U', 'U', 'U', 'C',
             'C', 'U', 'U', 'C', 'p', 'C', 'hmode', 'C', 'U', 'U', 'C', 'U', 'U', 'U', 'U', 'U', 'C', 'C', 'U',
             'U', 'U', 'U', 'U', 'C', 'C', 'U', 'U', 'U', 'C', 'input', 'input', 'p', 'p', 'p', 'p', 'p', 'p',
             'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['output'],
            name='selector_rewrite',
            domain='',
            equation='OLq,O,L,qks,qZs,qka,qja,qkb,qjb,qGA,qGy,yPP,Ade,qkd,qje,qHB,qHz,zRS,Bhg,qkg,qjh,DCM,D,CQp,D,CNi,qkN,qji,Clo,qkl,qjo,qIE,qIx,qIK,TUK,Etu,qkt,qju,qJF,qJx,qJK,VWK,Fwv,qkv,qjw,mkx,xXY,ncrk,nfrj,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p->ncrj',
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
            _tensor('p', TensorProto.FLOAT, (2,), INIT_P_0),
            _tensor('U', TensorProto.FLOAT, (2, 30, 2), INIT_U_1),
            _tensor('C', TensorProto.FLOAT, (2, 2, 2), INIT_C_2),
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
