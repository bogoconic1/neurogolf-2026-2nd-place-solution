from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task009'
TASK_NUM = 9
KAGGLE = {'score': 19.276415, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 306
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task009_expand_simplify_recompress'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task009_recompressed_dedup'
OPSETS = [('', 18)]


# S: FLOAT[10, 2], 20 value(s)
INIT_S_0 = [0.29635801911354065, -0.010393365286290646, 0.07067964971065521, 0.4562116861343384, 0.0217160414904356,
 0.3717886507511139, -0.02010669745504856, -0.529371440410614, 0.018208738416433334, 0.8434239625930786,
 -0.020903129130601883, 0.8093015551567078, 0.1697584092617035, 0.8905989527702332, 0.04391057789325714,
 0.34509578347206116, -0.09185279905796051, 0.42426037788391113, -0.002040209248661995, 0.34103840589523315]

# col_l: FLOAT[10, 3], 30 value(s)
INIT_COL_L_1 = [1.0377253293991089, 0.47485020756721497, 0.3155101239681244, 0.6228438019752502, 0.3723820149898529,
 -0.26649579405784607, -0.12416066229343414, 1.205091953277588, 0.11011108011007309, 0.5252043008804321,
 -0.3130803108215332, 0.7028577923774719, 0.8928139805793762, 0.026347462087869644, -0.017749711871147156,
 0.195044606924057, 0.6275696158409119, -0.14681251347064972, -0.19884751737117767, 0.25926101207733154,
 0.6890878677368164, 0.6733025312423706, -0.19039228558540344, 0.2541199028491974, -0.373495876789093,
 0.8391440510749817, 0.661170482635498, 0.11494190245866776, -0.14166493713855743, 1.2208726406097412]

# eq2: FLOAT[2, 2, 2], 8 value(s)
INIT_EQ2_2 = [0.6973984837532043, 0.18534834682941437, 0.0283956378698349, 1.0097739696502686, 0.3362899720668793,
 0.2416829764842987, 10.124098777770996, 1.1111781597137451]

# G: FLOAT[2, 2], 4 value(s)
INIT_G_3 = [0.1350724995136261, 18.186607360839844, 18.243452072143555, -0.10042431205511093]

# hi2: FLOAT[30, 2], 60 value(s)
INIT_HI2_4 = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# mid2: FLOAT[30, 2], 60 value(s)
INIT_MID2_5 = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# low3: FLOAT[30, 3], 90 value(s)
INIT_LOW3_6 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0010000000474974513, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 0.0010000000474974513, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0010000000474974513, 1.0, 0.0, 0.0, 1.0, 0.0,
 0.0, 0.0010000000474974513, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0010000000474974513, 0.0, 0.0, 0.0, 1.0,
 0.0, 0.0, 1.0, 0.0, 0.0, 0.0010000000474974513, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0010000000474974513, 0.0, 0.0, 0.0,
 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0010000000474974513, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
 0.0010000000474974513, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0010000000474974513, 0.0, 0.0]

# M: FLOAT[2, 2, 2, 2], 16 value(s)
INIT_M_7 = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]

# L: FLOAT[2, 3, 3], 18 value(s)
INIT_L_8 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]


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
        # 0: Einsum inputs=69 outputs=1
        _node(
            'Einsum',
            ['G', 'G', 'M', 'hi2', 'mid2', 'M', 'low3', 'L', 'M', 'M', 'L', 'low3', 'mid2', 'hi2', 'input',
             'input', 'S', 'S', 'col_l', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'eq2', 'eq2', 'L', 'L', 'S', 'S',
             'col_l', 'eq2', 'eq2', 'G', 'eq2', 'eq2', 'G', 'L', 'L', 'col_l', 'S', 'S', 'L', 'L', 'col_l', 'S',
             'S', 'low3', 'hi2', 'mid2', 'input', 'mid2', 'hi2', 'low3', 'M', 'M', 'L', 'low3', 'hi2', 'M',
             'mid2', 'M', 'L', 'input', 'low3', 'hi2', 'mid2'],
            ['output'],
            name='task009_tied_high_mid',
            domain='',
            equation='ee,ee,efgh,og,ol,fklj,on,knm,eMhK,MNjJ,NmL,IL,IJ,IK,...qrI,...qro,qt,qs,qu,pp,pe,pe,ep,ep,pp,pe,pbs,pct,edu,eud,ac,ab,ad,ecx,ebw,YY,YVb,YWc,YZ,ZdX,ZXd,UX,UV,UW,pdy,pyd,vy,vx,vw,im,ih,ij,...vzi,zB,zA,zC,pFAH,FDBG,DCE,rE,rH,pOHP,rG,OTGR,TES,...vQi,QS,QP,QR->...Uri',
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
            _tensor('S', TensorProto.FLOAT, (10, 2), INIT_S_0),
            _tensor('col_l', TensorProto.FLOAT, (10, 3), INIT_COL_L_1),
            _tensor('eq2', TensorProto.FLOAT, (2, 2, 2), INIT_EQ2_2),
            _tensor('G', TensorProto.FLOAT, (2, 2), INIT_G_3),
            _tensor('hi2', TensorProto.FLOAT, (30, 2), INIT_HI2_4),
            _tensor('mid2', TensorProto.FLOAT, (30, 2), INIT_MID2_5),
            _tensor('low3', TensorProto.FLOAT, (30, 3), INIT_LOW3_6),
            _tensor('M', TensorProto.FLOAT, (2, 2, 2, 2), INIT_M_7),
            _tensor('L', TensorProto.FLOAT, (2, 3, 3), INIT_L_8),
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
