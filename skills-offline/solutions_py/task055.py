from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task055'
TASK_NUM = 55
KAGGLE = {'score': 19.615505, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 218
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task055_m0_p218'
OPSETS = [('', 18)]


# qbasis: FLOAT[2, 30], 60 value(s)
INIT_QBASIS_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]

# HK: FLOAT[2, 2, 2], 8 value(s)
INIT_HK_1 = [0.25, 0.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0]

# ZK: FLOAT[4, 2, 2], 16 value(s)
INIT_ZK_2 = [50.909423828125, -0.35812631249427795, 0.11937301605939865, 0.00029358110623434186, -2.4171032905578613,
 -1.2085589170455933, 1.2085516452789307, 5.693663752026623e-06, 81.47982788085938, 0.00019833043916150928,
 -6.997653690632433e-05, -0.0004261245485395193, 37.972286224365234, 0.2670164704322815, -0.08900729566812515,
 0.00021866905444767326]

# UA: FLOAT[7, 4], 28 value(s)
INIT_UA_3 = [6.046584606170654, 0.5306550860404968, 8.754491806030273, -8.501992225646973, 1.3185914754867554, 0.2105766385793686,
 1.9088826179504395, -1.8533406257629395, -0.0028658853843808174, -0.0005035356152802706, -0.0027921097353100777,
 0.0013943193480372429, -0.00019447598606348038, -0.0002196477580582723, -0.0002817763015627861, 0.00027357274666428566,
 -0.5545240044593811, -0.01972758024930954, -0.8029443621635437, 0.7798854112625122, -0.22748583555221558,
 -0.0007853303104639053, -0.32890743017196655, 0.3191726803779602, 1.0453100003360305e-05, -9.271082603845571e-07,
 -4.48354403488338e-05, -1.4659948647022247e-05]

# VA: FLOAT[7, 4], 28 value(s)
INIT_VA_4 = [0.3670666515827179, 0.03186218813061714, 0.5314655303955078, -0.5161483287811279, -0.20003393292427063,
 -0.026568150147795677, -0.28968265652656555, 0.28135058283805847, 0.001643642783164978, -0.008809670805931091,
 0.002372145652770996, -0.0023077428340911865, 0.013681279495358467, 0.0016631208127364516, 0.007624924182891846,
 0.0044207945466041565, 0.15689820051193237, 0.0030622920021414757, 0.22716045379638672, -0.22067467868328094,
 0.20927764475345612, 0.008836529217660427, 0.30301225185394287, -0.294283390045166, 1.0453100003360305e-05,
 -9.271082603845571e-07, -4.48354403488338e-05, -1.4659948647022247e-05]

# O: FLOAT[7, 10], 70 value(s)
INIT_O_5 = [-1.0010000467300415, 2.495979070663452, 7.6915717124938965, 33.34231948852539, 23.576427459716797, 0.0,
 -12.731498718261719, 0.0, -0.0, 0.0, -1.0, -56.15019989013672, -64.57296752929688, -514.1255493164062,
 -229.73573303222656, 0.0, 218.97012329101562, 0.0, -0.0, 0.0, -1.0, -4388.16162109375, 6381.30859375,
 -369.3385314941406, -151.3135986328125, 0.0, 214.95101928710938, 0.0, -0.0, 0.0, 1.0010000467300415, 108.1194076538086,
 160.76312255859375, -209739.765625, 107780.7734375, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 13.176139831542969,
 -0.26950401067733765, 3028.21337890625, 1517.329345703125, 0.0, 577.6458129882812, 0.0, 0.0, 0.0, 1.0,
 446.1610412597656, 592.8992309570312, -161.85020446777344, -120.31324005126953, 0.0, 366.389892578125, 0.0, 0.0, 0.0,
 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0]

# AR: FLOAT[2, 4], 8 value(s)
INIT_AR_6 = [18.627670288085938, -5.853738067190806e-12, -10.05003833770752, -23.835432052612305, 1.8203500076197088e-05,
 -1.8500008583068848, 4.539775909506716e-05, -2.338173726457171e-05]


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
        # 0: Einsum inputs=33 outputs=1
        _node(
            'Einsum',
            ['O', 'input', 'qbasis', 'O', 'input', 'qbasis', 'qbasis', 'HK', 'AR', 'ZK', 'HK', 'HK', 'qbasis',
             'qbasis', 'input', 'O', 'qbasis', 'input', 'O', 'qbasis', 'ZK', 'qbasis', 'UA', 'ZK', 'HK', 'AR',
             'VA', 'ZK', 'qbasis', 'qbasis', 'O', 'input', 'O'],
            ['output'],
            name='',
            domain='',
            equation='aA,nABw,SB,bD,nDEw,qE,RE,RSq,Rp,pSx,xyy,UVt,tL,UL,nJhL,fJ,VI,nGhI,eG,dh,pcd,ch,Tp,PVr,ruu,UP,TP,Pjk,kw,jw,Tm,nmhw,To->nohw',
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
            _tensor('qbasis', TensorProto.FLOAT, (2, 30), INIT_QBASIS_0),
            _tensor('HK', TensorProto.FLOAT, (2, 2, 2), INIT_HK_1),
            _tensor('ZK', TensorProto.FLOAT, (4, 2, 2), INIT_ZK_2),
            _tensor('UA', TensorProto.FLOAT, (7, 4), INIT_UA_3),
            _tensor('VA', TensorProto.FLOAT, (7, 4), INIT_VA_4),
            _tensor('O', TensorProto.FLOAT, (7, 10), INIT_O_5),
            _tensor('AR', TensorProto.FLOAT, (2, 4), INIT_AR_6),
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
