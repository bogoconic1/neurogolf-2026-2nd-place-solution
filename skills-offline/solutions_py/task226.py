from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task226'
TASK_NUM = 226
KAGGLE = {'score': 20.002788, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 148
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'gpt-5.6-sol/swarm-20plus'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task226_gpt56sol_tensor_network'
OPSETS = [('', 18)]


# F: FLOAT[30, 2], 60 value(s)
INIT_F_0 = [-0.35258281230926514, -0.2272416055202484, -0.21651071310043335, -0.13800157606601715, -0.32650133967399597,
 -0.12663966417312622, 0.3185528516769409, 0.055503614246845245, -0.39082732796669006, 0.12118937075138092,
 -0.3207102119922638, 0.10654539614915848, -0.6477823257446289, 0.4860590994358063, -0.906256914138794,
 0.7064106464385986, 0.8410412669181824, -0.6662518978118896, -1.2901427745819092, 1.0220946073532104, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# P: FLOAT[3, 2], 6 value(s)
INIT_P_1 = [0.44773030281066895, 0.2860602140426636, 0.3486638069152832, 0.6218308806419373, 0.663128674030304, 0.70343017578125]

# C: FLOAT[4, 10], 40 value(s)
INIT_C_2 = [-3.5691111087799072, 0.4691416323184967, 1.4169111251831055, 2.7164013385772705, 13.781420707702637, 35.15550231933594,
 12.859570503234863, 13.189373016357422, 12.953740119934082, 13.229148864746094, 80.16072082519531, 2373.060302734375,
 4147.93359375, 100.75997924804688, 2679.35400390625, -56.934654235839844, 3098.577392578125, 2890.676513671875,
 3134.6943359375, 2835.21875, 6.392612934112549, -1.411586046218872, -4.409534454345703, 2.7431938648223877,
 3.114100933074951, 0.6093409061431885, 3.2200498580932617, 3.212376117706299, 3.094264268875122, 3.1606619358062744,
 -11.327962875366211, 9.66600227355957, -2.0863306522369385, 9.672685623168945, 3.2236037254333496, -1.481091856956482,
 3.7069127559661865, 3.5031139850616455, 3.1824400424957275, 3.2071285247802734]

# B: FLOAT[4, 3], 12 value(s)
INIT_B_3 = [0.23193426430225372, 0.003844053018838167, 0.2734101712703705, -0.37646159529685974, -0.04117467999458313,
 0.056768666952848434, 0.5264042615890503, -0.32101762294769287, -0.37416452169418335, -0.01998552866280079,
 -0.03985186293721199, 0.14979325234889984]

# U: FLOAT[3, 3], 9 value(s)
INIT_U_4 = [0.4776941239833832, -0.42085665464401245, -0.06280523538589478, -0.017711542546749115, 0.8028596639633179,
 -1.0246413946151733, -0.6364673972129822, -0.08130563795566559, 0.5511281490325928]

# A: FLOAT[3, 3], 9 value(s)
INIT_A_5 = [-0.22723029553890228, 0.5107736587524414, 0.9796286225318909, 0.6627377867698669, -0.989082932472229,
 0.8729957342147827, -0.23839840292930603, -0.3855847716331482, -0.7091676592826843]

# T: FLOAT[4, 3], 12 value(s)
INIT_T_6 = [-0.6411707401275635, 0.7030643224716187, 0.29316291213035583, -0.06084366887807846, 0.14117158949375153,
 0.20105668902397156, 0.9184787273406982, -0.5455424785614014, 0.04978911578655243, -0.0626065582036972,
 1.3108717203140259, -1.162835955619812]


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
            ['B', 'T', 'C', 'B', 'T', 'C', 'input', 'input', 'F', 'F', 'P', 'P', 'B', 'T', 'C', 'B', 'T', 'C',
             'input', 'input', 'F', 'F', 'P', 'P', 'U', 'A', 'T', 'U', 'A', 'T', 'U', 'A', 'T', 'P', 'P', 'P',
             'P', 'F', 'F', 'B', 'T', 'C', 'input', 'F', 'F', 'C'],
            ['output'],
            name='',
            domain='',
            equation='gI,LI,gp,QJ,fJ,fl,blys,bpqs,qA,qB,iA,iB,mK,ZK,mx,VN,nN,nd,bdvr,bxvz,zE,zF,jE,jF,iR,aR,tR,jS,eS,tS,cU,cU,tU,aC,aD,eG,eH,hC,hD,tW,YW,Yk,bkhw,wG,wH,to->bohw',
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
            _tensor('P', TensorProto.FLOAT, (3, 2), INIT_P_1),
            _tensor('C', TensorProto.FLOAT, (4, 10), INIT_C_2),
            _tensor('B', TensorProto.FLOAT, (4, 3), INIT_B_3),
            _tensor('U', TensorProto.FLOAT, (3, 3), INIT_U_4),
            _tensor('A', TensorProto.FLOAT, (3, 3), INIT_A_5),
            _tensor('T', TensorProto.FLOAT, (4, 3), INIT_T_6),
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
