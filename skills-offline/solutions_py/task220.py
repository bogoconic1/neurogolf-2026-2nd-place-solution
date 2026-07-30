from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task220'
TASK_NUM = 220
KAGGLE = {'score': 20.40488, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 99
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task220_repeated_power_factor_p99'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task220_m0_p99'
OPSETS = [('', 13)]


# H: FLOAT[30, 2], 60 value(s)
INIT_H_0 = [1.0013386011123657, -0.0005430012242868543, 0.9825366139411926, 0.18363749980926514, 0.931479275226593,
 0.3610747754573822, 0.8506963849067688, 0.5257640480995178, 0.7397484183311462, 0.6730949282646179, 0.6033411026000977,
 0.7970927953720093, 0.44687262177467346, 0.8945973515510559, 0.2753605842590332, 0.9613332152366638,
 0.0940961241722107, 0.9958506226539612, -0.09015462547540665, 0.9952057600021362, -0.2710746228694916,
 0.9629707932472229, -0.4438013732433319, 0.8964712619781494, -0.5998781323432922, 0.7980378270149231,
 -0.7367793321609497, 0.6756409406661987, -0.8490716814994812, 0.5294308662414551, -0.9313124418258667,
 0.3642464876174927, 0.0003710431628860533, -5.42876441613771e-05, -0.00037622038507834077, -0.00045147145283408463,
 -1.6101614164654166e-05, 2.0838127966271713e-05, -2.564973692642525e-05, -2.4033530280576088e-05,
 8.000896195881069e-05, 0.00018892515799961984, -5.3593204938806593e-05, 2.3933753254823387e-05, -0.00025674540665932,
 -1.968912147276569e-05, 1.000917109195143e-05, -0.00010779743024613708, -0.0003388073237147182,
 -0.00011137628462165594, -0.0001017090689856559, 5.0008471589535475e-05, 0.00011545691813807935,
 -0.00022530974820256233, 6.921905878698453e-05, -0.0002052619238384068, -0.00020600289280992, -0.0003812970535364002,
 -0.00014526814629789442, 0.00010287740587955341]

# U: FLOAT[10, 3], 30 value(s)
INIT_U_1 = [-1.8748743534088135, 0.44326892495155334, 0.9806384444236755, -0.8035060167312622, 2.129249095916748,
 0.9798913598060608, -1.895972490310669, -0.964115560054779, 4.4883036613464355, -1.6121892929077148,
 -5.609190464019775, -0.29177945852279663, -0.3060733377933502, 2.374150037765503, 2.7612688541412354,
 -1.1366093158721924, 0.04991525039076805, 0.7716427445411682, 2.0370001792907715, -0.8218220472335815,
 0.3285670876502991, -0.6493949890136719, 1.2582523822784424, -0.28335586190223694, -1.5845750570297241,
 0.7389448881149292, -4.9115142822265625, 1.539982795715332, 0.07241404801607132, 0.03934395685791969]

# W: FLOAT[3, 3], 9 value(s)
INIT_W_2 = [-24.543058395385742, 126.19763946533203, -75.39318084716797, 23.21442222595215, -30.416688919067383, 7.809772968292236,
 -5.453826904296875, -45.339195251464844, 52.24383544921875]


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
        # 0: Einsum inputs=92 outputs=1
        _node(
            'Einsum',
            ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'input', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'H', 'H', 'H', 'H', 'U', 'U', 'W', 'input', 'U', 'W', 'U', 'U', 'U'],
            ['output'],
            name='output',
            domain='',
            equation='hg,rg,hi,ri,hj,rj,hk,rk,hl,rl,hm,rm,hn,rn,hp,rp,hq,rq,ht,rt,hu,ru,hv,rv,hx,rx,hy,ry,hz,rz,hA,rA,hB,rB,hC,rC,hD,rD,hE,rE,...ers,wF,sF,wG,sG,wH,sH,wI,sI,wJ,sJ,wK,sK,wL,sL,wM,sM,wN,sN,wO,sO,wP,sP,wQ,sQ,wR,sR,wS,sS,wT,sT,wU,sU,wV,sV,wW,sW,wX,sX,wY,sY,wZ,sZ,eb,ec,cf,...dhw,da,ab,oa,ob,of->...ohw',
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
            _tensor('H', TensorProto.FLOAT, (30, 2), INIT_H_0),
            _tensor('U', TensorProto.FLOAT, (10, 3), INIT_U_1),
            _tensor('W', TensorProto.FLOAT, (3, 3), INIT_W_2),
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
