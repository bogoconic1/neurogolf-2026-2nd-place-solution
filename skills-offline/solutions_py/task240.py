from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task240'
TASK_NUM = 240
KAGGLE = {'score': 20.24641, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 116
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task240_p116_deg9_foldD'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task240_p116_deg9_foldD'
OPSETS = [('', 17)]


# F: FLOAT[30, 2], 60 value(s)
INIT_F_0 = [1.1383779048919678, 0.6277704834938049, 0.6719790101051331, 1.1128541231155396, 1.1383779048919678, 0.6277704834938049,
 0.025523798540234566, 1.2997493743896484, 1.1383779048919678, 0.6277704834938049, -0.6277704834938049,
 1.1383779048919678, 1.1383779048919678, 0.6277704834938049, -1.1128541231155396, 0.6719790101051331,
 1.1383779048919678, 0.6277704834938049, -1.2997493743896484, 0.025523798540234566, 1.1383779048919678,
 0.6277704834938049, -1.1128541231155396, 0.6719790101051331, 1.1383779048919678, 0.6277704834938049,
 -0.6277704834938049, 1.1383779048919678, 1.1383779048919678, 0.6277704834938049, 0.025523798540234566,
 1.2997493743896484, 1.1383779048919678, 0.6277704834938049, 0.6719790101051331, 1.1128541231155396, 1.1383779048919678,
 0.6277704834938049, -0.8135150074958801, 1.8043556213378906, 1.2982147932052612, -1.6696374416351318,
 0.41344335675239563, -1.693505048751831, 1.6862914562225342, -1.1974021196365356, -1.8012590408325195,
 -0.8284426331520081, -1.154641032218933, -1.559841275215149, 0.9424543380737305, 1.6836920976638794,
 -0.2209792137145996, -1.6514153480529785, -1.6976584196090698, -1.1626397371292114, -1.6644349098205566,
 1.4241180419921875, 1.8710095882415771, 1.0803022384643555]

# A: FLOAT[2, 2, 7], 28 value(s)
INIT_A_1 = [4.165689824731089e-05, -0.0935911312699318, -1.8301113843917847, -0.6259056329727173, 0.6237479448318481,
 -0.006665501277893782, -0.6184142827987671, -0.00013437795860227197, -0.009062794037163258, 2.329705238342285,
 2.6233901977539062, -0.542241096496582, 0.011753624305129051, 0.6860002875328064, 4.07212100981269e-05,
 -0.16112273931503296, -2.2896502017974854, -0.33854612708091736, 0.8501705527305603, -0.007267866749316454,
 -0.8068084120750427, -6.913379911566153e-05, 0.14483729004859924, 0.6688417792320251, -2.5338010787963867,
 -0.40648677945137024, 0.0002560921129770577, 0.29881471395492554]

# C: FLOAT[7, 2, 2], 28 value(s)
INIT_C_2 = [-8.199240684509277, 0.44687551259994507, 6.520003795623779, 1.329168677330017, -2.277282238006592, -0.8931547999382019,
 0.7018331289291382, -0.6967054009437561, -0.883508563041687, 0.9693087935447693, 1.7120084762573242,
 0.5528594851493835, -0.7256103157997131, -0.7448448538780212, 0.8620704412460327, -0.2777860760688782,
 -0.8598714470863342, 1.2498019933700562, 0.4176817834377289, 0.067307248711586, -5.171120643615723,
 -1.1538830995559692, 3.4841811656951904, -0.37659719586372375, -1.1319799423217773, -0.4656711518764496,
 0.6138585805892944, -1.0089589357376099]


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
        # 0: Einsum inputs=71 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'A', 'C', 'C', 'C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'C', 'C',
             'C', 'A', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'C',
             'input', 'F', 'F', 'F', 'F', 'F', 'C', 'C', 'C', 'A', 'C', 'C', 'C', 'C', 'C', 'F', 'F', 'F', 'F',
             'F', 'A', 'C', 'C', 'C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'input'],
            ['output'],
            name='task240_p116_deg9_foldD',
            domain='',
            equation='ea,ea,ea,ea,eb,eb,ed,abr,rZp,rZl,rZo,rZm,rZk,ho,hl,hk,hp,hm,tdb,tVX,TXV,XVT,UV,UV,UX,UX,UX,UX,UX,UX,Uu,XuT,XuT,uvs,sYD,sYE,sYB,sYq,sYA,nchw,wD,wA,wB,wE,wq,tRS,tij,tfg,fgx,xKI,xKC,xKG,xKF,xKJ,HC,HI,HF,HJ,HG,ijy,yQM,yQL,yQN,yQO,yQP,WO,WM,WN,WL,WP,nzHW->ncHW',
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
            _tensor('F', TensorProto.FLOAT, (30, 2), INIT_F_0),
            _tensor('A', TensorProto.FLOAT, (2, 2, 7), INIT_A_1),
            _tensor('C', TensorProto.FLOAT, (7, 2, 2), INIT_C_2),
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
