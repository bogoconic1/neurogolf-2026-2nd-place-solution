from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task221'
TASK_NUM = 221
KAGGLE = {'score': 20.14797, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 128
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task221-derived-theta'
OPSETS = [('', 18)]


# Coord: FLOAT[30, 3], 90 value(s)
INIT_COORD_0 = [1.6177400350570679, 0.3905010223388672, 0.44089990854263306, 1.601296305656433, 0.37717652320861816,
 -0.39972633123397827, 1.7242097854614258, -0.6936533451080322, 0.00872467365115881, 0.7624519467353821,
 1.7198940515518188, 0.7713938355445862, 0.7991384267807007, 1.7749478816986084, -0.8087359070777893,
 0.8271793127059937, -1.8072381019592285, 0.8222266435623169, 0.6663666367530823, 1.9924001693725586,
 0.8433111310005188, 0.6939038634300232, 2.062546491622925, -0.8683510422706604, 0.7115908861160278,
 -2.1567115783691406, 0.9229788184165955, 0.4197838306427002, 1.7224026918411255, 1.0887893438339233, 0.423904687166214,
 1.730391025543213, -1.1003210544586182, 0.4299381673336029, -1.6072018146514893, 1.1167376041412354,
 -0.13337185978889465, 1.7779192924499512, 1.657199501991272, -0.15010753273963928, 1.8210299015045166,
 -1.7019124031066895, 0.15126557648181915, -1.8368626832962036, 1.7169262170791626, -0.0268352460116148,
 1.4142160415649414, 1.9741960763931274, -0.027832448482513428, 1.4399131536483765, -2.0067684650421143,
 0.03149067610502243, -1.440200924873352, 2.0081045627593994, -0.01521213911473751, 0.7840420007705688,
 2.0979981422424316, -0.04815754294395447, 0.7776172757148743, -2.11618971824646, 0.05185000225901604,
 -0.779559314250946, 2.1261680126190186, -4.264041900634766, 4.264041900634766, 0.0, -4.204578399658203,
 4.204578399658203, 0.0, 0.4670558273792267, -0.4670558273792267, 0.0, 0.0, -5.3029866218566895, 5.3029866218566895,
 0.0, -2.7502918243408203, 2.7502918243408203, 0.0, -0.44408169388771057, 0.44408169388771057, -5.027249336242676, 0.0,
 5.027249336242676, -3.6776859760284424, 0.0, 3.6776859760284424, 7.116593837738037, 0.0, -7.116593837738037]

# Feat: FLOAT[10, 2], 20 value(s)
INIT_FEAT_1 = [0.004272035788744688, 0.8764159083366394, 0.8141701817512512, -0.1710924357175827, 0.7653559446334839,
 -0.20149879157543182, 0.7954545021057129, -0.17467080056667328, 0.7810553908348083, -0.1832652986049652,
 0.8439362049102783, -0.18599484860897064, 0.7809807658195496, -0.18337324261665344, 0.7873831391334534,
 -0.17809787392616272, 0.790320873260498, -0.17927667498588562, 0.7962561249732971, -0.17572911083698273]

# Pair: FLOAT[2, 2, 3], 12 value(s)
INIT_PAIR_2 = [0.24362173676490784, -0.005052454769611359, 0.44796425104141235, 0.2450794279575348, 0.31631895899772644,
 -0.3218793272972107, 1.357039213180542, -0.1705397665500641, 0.0982780009508133, -0.465832382440567,
 -0.5257186889648438, -0.15066851675510406]

# Adapter: FLOAT[3, 2], 6 value(s)
INIT_ADAPTER_3 = [-0.882196843624115, -0.1756182461977005, 75.83616638183594, 0.11179795861244202, 2.2345468997955322, 3.892151117324829]


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
        # 0: Einsum inputs=74 outputs=1
        _node(
            'Einsum',
            ['input', 'Feat', 'Coord', 'Coord', 'Coord', 'Coord', 'Coord', 'Coord', 'Coord', 'Coord', 'Coord',
             'Coord', 'input', 'Feat', 'input', 'Feat', 'Coord', 'Coord', 'Coord', 'Coord', 'Pair', 'Pair',
             'Coord', 'Coord', 'Coord', 'input', 'Feat', 'Pair', 'Pair', 'Adapter', 'Pair', 'Pair', 'Coord',
             'Coord', 'Coord', 'Coord', 'Coord', 'Coord', 'Pair', 'Pair', 'Pair', 'Pair', 'Pair', 'Pair',
             'Pair', 'Pair', 'Pair', 'Pair', 'Pair', 'Pair', 'Pair', 'Pair', 'Pair', 'Pair', 'Feat', 'Feat',
             'Feat', 'Feat', 'Feat', 'Pair', 'Pair', 'Pair', 'Pair', 'Pair', 'Pair', 'Pair', 'Pair', 'Pair',
             'Pair', 'Pair', 'Pair', 'Pair', 'Pair', 'Pair'],
            ['output'],
            name='',
            domain='',
            equation='bcij,cS,ia,ha,iA,rA,hA,jd,wd,jD,sD,wD,bqBC,qm,btHI,tn,he,he,wf,wf,mne,mnf,pf,pf,pf,boUV,oT,mng,STg,gT,STk,MLk,Rk,Rk,hx,hy,wz,wv,lLu,lLu,ELF,ELF,GLJ,GLJ,GLJ,GLJ,GLJ,GLJ,KLN,KLN,KLN,KLN,KLN,KLN,OL,OL,OL,OL,OL,LPQ,LPQ,LPQ,LPQ,LWX,LWX,LWX,LWX,LWX,LWX,LWX,LWX,LWX,LWX,LWX->bohw',
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
            _tensor('Coord', TensorProto.FLOAT, (30, 3), INIT_COORD_0),
            _tensor('Feat', TensorProto.FLOAT, (10, 2), INIT_FEAT_1),
            _tensor('Pair', TensorProto.FLOAT, (2, 2, 3), INIT_PAIR_2),
            _tensor('Adapter', TensorProto.FLOAT, (3, 2), INIT_ADAPTER_3),
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
