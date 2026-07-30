from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task351'
TASK_NUM = 351
KAGGLE = {'score': 19.774253, 'date': '2026-07-14'}
MEMORY_BYTES = 86
PARAMS = 100
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'codex-task351-four-lane'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task351_fixed_order_four_lane'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# CASES: STRING[29], 29 value(s)
INIT_CASES_1 = ['9223370936701272065,-4481119014784892927,-36688503995645951,-4611686018427404287',
 '-333266372426481663,-1731783796623687679,9205357088589430785,-73190095311618047',
 '-2418741138065342463,-4785216449674297343,-16383,-1152921521803526143',
 '-35485433855,-307627524095,-69222399,-6931462038988275711',
 '8068057440086048769,-4611686035758301183,-1130297953370111,-288230448101081087',
 '-6665758459348140031,-53880918589439,-3448618220535807,6908521828386324481',
 '-18726638469119,6908521828385275905,-4899916394646241279,-180284742061867007',
 '6906270028572639233,-112730728173682687,-1729691220231536639,-4899916948665548799',
 '-3464398411401822207,8997873197114179585,-5478769684201357311,-36109061904678911',
 '-295547486207,-4785255530119167,-2335431418003455,-1125904222814207',
 '9151314438521864193,-2902654081212891135,-4621257268258750463,-34360803327',
 '-2252349569515519,-4900481581283557375,-69024137215,9205353218822619137',
 '-4899916416054214655,-2252350648500223,-277159624703,-2315434084083580927',
 '-36320167736655871,-175999438323711,-19705468934963199,9223371862907928577',
 '-536887295,-1152921521804050431,-7031244918232203263,-288114589695',
 '-1157495490158280703,9222809086867095553,-15846113279,-4647761063725514751',
 '-18051781904842751,-4612337205262958591,-74731606450913279,-2747195824269361151',
 '8551126229518893057,-2314885538886270975,-18730360782847,-1077968895',
 '-1724034500804607,-38584114258919423,-2738200942947090431,-72057731476914175',
 '8056939733365669889,-288247968338034687,-4612814208103399423,-38878733305675775',
 '-562950130728959,8214424981761671169,-29299957655683071,-2377900603253899263',
 '-288230378349543423,-1308304487844741119,-6935618192945463295,8646873899939643393',
 '9223371949071761409,-76561193666363391,-146950974592794623,-1154399248234594303',
 '-33587199,-8507299696102883327,-38577742266515455,-577200724167901183',
 '-206161592319,-1460294653076062207,-9895877820415,-831195606377725951',
 '-2193637375,-413863002111,-1099511644159,-133699174399',
 '9223372019674628097,-5044031582654971903,-1780112145137663,-1191624932648304639',
 '-72075238743359487,-16383,-211612172287,9223370928744808449',
 '-4611712475426013183,-1244405275589378047,9214364829009035265,-578713651649724415']

# P: INT64[30], 30 value(s)
INIT_P_2 = [1, 2, 4, 8, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# D: INT64[4, 10], 40 value(s)
INIT_D_3 = [0, 1, 33554432, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 33554432, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 33554432, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 1, 33554432]


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
            name='rng_f',
            domain='',
            dtype=10,
            high=13.812538146972656,
            low=-16.865692138671875,
            seed=559349184.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['rng_f'],
            ['case_i'],
            name='case_i',
            domain='',
            to=6,
        ),
        # 2: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['CASES', 'case_i'],
            ['case_string'],
            name='case_string',
            domain='',
            axis=0,
        ),
        # 3: StringSplit inputs=1 outputs=2
        _node(
            'StringSplit',
            ['case_string'],
            ['parts', 'lengths'],
            name='parts',
            domain='',
            delimiter=',',
            maxsplit=3,
        ),
        # 4: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['parts'],
            ['packed'],
            name='packed',
            domain='',
            to=7,
        ),
        # 5: Einsum inputs=8 outputs=1
        _node(
            'Einsum',
            ['packed', 'D', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['output'],
            name='output',
            domain='',
            equation='bk,kc,r,r,r,r,r,w->bcrw',
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
            _tensor('LIKE', TensorProto.FLOAT16, (1,), INIT_LIKE_0),
            _tensor('CASES', TensorProto.STRING, (29,), INIT_CASES_1),
            _tensor('P', TensorProto.INT64, (30,), INIT_P_2),
            _tensor('D', TensorProto.INT64, (4, 10), INIT_D_3),
        ],
        value_info=[
            _vi('rng_f', TensorProto.FLOAT16, [1]),
            _vi('case_i', TensorProto.INT32, [1]),
            _vi('case_string', TensorProto.STRING, [1]),
            _vi('parts', TensorProto.STRING, [1, 4]),
            _vi('lengths', TensorProto.INT64, [1]),
            _vi('packed', TensorProto.INT64, [1, 4]),
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
