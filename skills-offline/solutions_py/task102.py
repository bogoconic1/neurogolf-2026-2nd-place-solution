from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task102'
TASK_NUM = 102
KAGGLE = {'score': 19.852506, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 172
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'noH'
OPSETS = [('', 18)]


# B: FLOAT[30, 4], 120 value(s)
INIT_B_0 = [-1.2753230333328247, 0.7514790892601013, 0.18254108726978302, -0.18114648759365082, -0.8201218843460083,
 0.7268388271331787, -0.45831191539764404, -0.5320600271224976, -0.6297351717948914, 0.6053894758224487,
 -0.5228027105331421, -0.40632569789886475, -0.47941523790359497, 0.5335045456886292, -0.7997645735740662,
 -0.08287616074085236, -0.13090020418167114, 0.09976798295974731, -0.789997398853302, 1.0802634954452515,
 0.35430803894996643, -0.2391662746667862, -0.8106321692466736, 1.0128318071365356, -0.3945297300815582,
 0.03448544442653656, -0.054392844438552856, 0.22075678408145905, -1.0207741260528564, 0.2120257318019867,
 -0.14799842238426208, 0.8802089691162109, -0.10880530625581741, -0.08161807805299759, -0.7985964417457581,
 -0.7991349697113037, -0.5373904705047607, 0.06031404063105583, -0.4093311131000519, -1.1437057256698608,
 0.06258935481309891, -0.17044493556022644, -0.9500304460525513, -0.5217121243476868, -0.299752801656723,
 -0.09154944121837616, -0.7769023180007935, 0.3855568766593933, 0.5883629322052002, -0.5915367603302002,
 -0.10405006259679794, 0.10690003633499146, -0.7012384533882141, -0.9170305728912354, 0.998425304889679,
 -0.3546867072582245, 0.003022184129804373, -0.2890773415565491, -0.3179304003715515, -0.20487524569034576,
 -0.04621739313006401, -0.3103349506855011, -0.3396325409412384, -0.1918608546257019, -0.03183632716536522,
 -0.31646692752838135, -0.3729308545589447, -0.2061217874288559, -0.020038781687617302, -0.9628190994262695,
 0.8733816146850586, -0.3802824914455414, -0.4697362780570984, -1.41154944896698, 1.599239706993103,
 -0.8065030574798584, 1.10636305809021, -0.8571901917457581, -0.20399916172027588, 0.2622203230857849,
 0.559293270111084, -0.5760519504547119, -0.612376868724823, -0.1357487291097641, 0.0011810854775831103,
 -0.28226715326309204, -0.30972951650619507, -0.20043154060840607, 0.9334679245948792, -1.543504238128662,
 0.5293260812759399, -0.6697533130645752, 0.9129542112350464, -0.75679612159729, 0.3918644189834595,
 -0.9814228415489197, 0.017732268199324608, -0.32461556792259216, -0.3678404986858368, -0.23071801662445068,
 0.7323452830314636, 0.5793341994285583, 0.2105141282081604, -0.7585818767547607, 0.8434746265411377,
 0.9577839374542236, 0.3379098176956177, 1.8613355159759521, 2.056868076324463, -1.8091862201690674,
 -0.3212757110595703, -0.7204205393791199, -1.499407172203064, -0.6589426398277283, -1.550661563873291,
 -0.19266514480113983, -0.2745417058467865, -0.19290098547935486, 0.9692336916923523, 0.15131185948848724]

# Ccore: FLOAT[2, 4, 4], 32 value(s)
INIT_CCORE_1 = [-0.000808432640042156, 0.15116089582443237, -0.17629490792751312, 0.17860907316207886, -0.008912281133234501,
 -0.035117585211992264, 0.1880534589290619, -0.45371949672698975, -0.029533568769693375, 0.20950482785701752,
 -0.18061451613903046, 0.15562684834003448, -0.013989640399813652, 0.14176137745380402, -0.05809371545910835,
 0.22677084803581238, -0.8414688110351562, 0.537256121635437, 0.3682374060153961, 0.700908899307251, -0.331201434135437,
 -0.5338068604469299, 0.4473452568054199, -0.07489442825317383, -0.8080516457557678, 0.6983284950256348,
 -0.05406291410326958, -0.5316652655601501, 0.0939159244298935, 0.3090038597583771, -1.2517979145050049,
 0.20442509651184082]

# O: FLOAT[2, 10], 20 value(s)
INIT_O_2 = [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


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
        # 0: Einsum inputs=27 outputs=1
        _node(
            'Einsum',
            ['B', 'input', 'B', 'O', 'B', 'B', 'B', 'Ccore', 'B', 'B', 'B', 'Ccore', 'B', 'B', 'B', 'Ccore',
             'Ccore', 'Ccore', 'B', 'B', 'B', 'Ccore', 'B', 'O', 'input', 'B', 'O'],
            ['output'],
            name='',
            domain='',
            equation='yC,...qxy,xA,gq,uC,uC,uJ,rJl,vA,vA,vM,rMl,wD,wD,wP,fPl,fUV,QWV,ZU,ZU,ZW,QEl,cD,fh,...hbc,bE,fa->...abc',
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
            _tensor('B', TensorProto.FLOAT, (30, 4), INIT_B_0),
            _tensor('Ccore', TensorProto.FLOAT, (2, 4, 4), INIT_CCORE_1),
            _tensor('O', TensorProto.FLOAT, (2, 10), INIT_O_2),
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
