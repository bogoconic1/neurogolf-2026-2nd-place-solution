from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task012'
TASK_NUM = 12
KAGGLE = {'score': 19.394198, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 272
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 7
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 12)]


# F: FLOAT[10, 3], 30 value(s)
INIT_F_0 = [1.2356432676315308, 0.9708110094070435, 2.200830948595467e-07, 1.2356432676315308, -0.9708110094070435,
 2.200830948595467e-07, 1.2356432676315308, 0.9708110094070435, -2.200830948595467e-07, 1.2356432676315308,
 -0.9708110094070435, -2.200830948595467e-07, 1.2356432676315308, 0.9708110094070435, 2.200830948595467e-07,
 1.2356432676315308, -0.9708110094070435, 2.200830948595467e-07, 1.2356432676315308, 0.9708110094070435,
 -2.200830948595467e-07, 1.2356432676315308, -0.9708110094070435, -2.200830948595467e-07, 1.2356432676315308,
 0.9708110094070435, 2.200830948595467e-07, 1.2356432676315308, -0.9708110094070435, 2.200830948595467e-07]

# G1: FLOAT[2, 3], 6 value(s)
INIT_G1_1 = [0.3565768003463745, 0.0, 0.0, 0.2674325704574585, 0.0, 8429991624704.0]

# T: FLOAT[10, 3], 30 value(s)
INIT_T_2 = [0.911722719669342, 1.013023018836975, 0.0, 0.911722719669342, 1.013023018836975, 0.0, 0.911722719669342,
 1.013023018836975, 0.0, 0.911722719669342, 1.013023018836975, 0.0, 0.911722719669342, -0.5065115094184875,
 700.4325561523438, 0.911722719669342, -0.5065115094184875, 700.4325561523438, 0.911722719669342, -0.5065115094184875,
 700.4325561523438, 0.911722719669342, -0.5065115094184875, 700.4325561523438, 0.911722719669342, -0.5065115094184875,
 -700.4325561523438, 0.911722719669342, -0.5065115094184875, -700.4325561523438]

# C3: FLOAT[2, 3], 6 value(s)
INIT_C3_3 = [1.2030243873596191, 0.0, 3.057444786236374e-08, 0.40100812911987305, 0.6496360301971436, 1.0191482715526945e-06]

# A: FLOAT[30, 5], 150 value(s)
INIT_A_4 = [1.1999160051345825, -1.9633198976516724, 1.4000179767608643, 1.4387881755828857, 1.0831915140151978,
 -1.6807398796081543, -1.3251605033874512, 1.2743501663208008, 1.8111252784729004, -1.481411099433899,
 2.306056261062622, -0.6361874938011169, -1.3961832523345947, 1.4972357749938965, 1.9199961423873901,
 -2.354843854904175, 0.28689146041870117, -1.3230446577072144, 1.7023879289627075, -1.8581089973449707,
 1.5766680240631104, 1.0079460144042969, 1.4296597242355347, 1.7071887254714966, 2.0927791595458984,
 -0.6309588551521301, 1.4760702848434448, 1.6312837600708008, 1.5257736444473267, -2.264631986618042,
 -0.6307127475738525, 1.565283179283142, -1.7093740701675415, 1.6171581745147705, 2.1617000102996826, 1.567662000656128,
 0.9952394962310791, -1.4365930557250977, 1.6708728075027466, -2.1123476028442383, -2.3655483722686768,
 0.2943165898323059, 1.3140751123428345, 1.722497582435608, 1.8499236106872559, 2.3013384342193604, -0.6369968056678772,
 1.4032269716262817, 1.5178576707839966, -1.940946102142334, -1.6661909818649292, -1.3351787328720093,
 -1.2752368450164795, 1.8458551168441772, 1.5368560552597046, 1.1839314699172974, -1.9668102264404297,
 -1.402184247970581, 1.4905797243118286, -1.094170093536377, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# BQL: FLOAT[3, 5], 15 value(s)
INIT_BQL_5 = [0.46338188648223877, 0.5513896346092224, 0.006226567085832357, 0.3533910810947418, 0.5627242922782898,
 -0.5228639841079712, 0.7813645005226135, 0.04240652173757553, 0.4830600321292877, -0.5580593347549438,
 -0.4213182032108307, -0.14739985764026642, 0.8774128556251526, -0.2603093385696411, -0.6016996502876282]

# RV: FLOAT[2, 5], 10 value(s)
INIT_RV_6 = [0.22568070888519287, 0.23859502375125885, 0.10346690565347672, 0.09489504992961884, -0.4216059744358063,
 -0.534417986869812, -0.2451273798942566, -0.20154231786727905, -0.4153543710708618, 0.36857834458351135]

# RM: FLOAT[2, 5], 10 value(s)
INIT_RM_7 = [0.43824806809425354, -0.06096010282635689, -0.182022824883461, 0.3003172278404236, 0.21419885754585266,
 -0.3800960183143616, 0.3365989923477173, 0.1348792016506195, -0.30221498012542725, -0.5174170732498169]

# R3: FLOAT[3, 5], 15 value(s)
INIT_R3_8 = [-0.0005146187613718212, -0.009214634075760841, 0.0012662606313824654, -0.007036857306957245, -0.0004880000778939575,
 18.884532928466797, 2.0752851963043213, 8.587347984313965, 7.537381649017334, 18.153932571411133, -7239456260096.0,
 -582168150016.0, -3321544048640.0, -2738436440064.0, -6957457473536.0]


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
        # 0: Einsum inputs=66 outputs=1
        _node(
            'Einsum',
            ['T', 'input', 'F', 'F', 'C3', 'C3', 'F', 'F', 'G1', 'C3', 'T', 'input', 'T', 'F', 'F', 'G1', 'F',
             'C3', 'C3', 'C3', 'R3', 'F', 'T', 'G1', 'BQL', 'RV', 'RM', 'T', 'F', 'input', 'F', 'C3', 'C3', 'F',
             'input', 'F', 'T', 'C3', 'C3', 'T', 'T', 'input', 'F', 'F', 'G1', 'F', 'BQL', 'F', 'A', 'A',
             'input', 'A', 'BQL', 'BQL', 'A', 'A', 'A', 'BQL', 'A', 'A', 'A', 'BQL', 'BQL', 'A', 'A', 'A'],
            ['output'],
            name='einsum_render',
            domain='',
            equation='aQ,...aAB,aP,aN,MQ,MN,cP,cN,MP,VT,cT,...bCD,bT,bS,bR,VS,cS,VR,Yn,mn,nk,cR,cQ,YX,pk,Vk,Mk,dZ,dX,...dEF,dW,YW,YZ,oX,...oJK,oW,oZ,mg,ml,ol,il,...iGH,og,ig,mj,ij,pe,oj,re,rL,...crs,rI,pL,pI,hI,he,hL,pU,sU,sf,sO,pO,pf,wf,wU,wO->...ohw',
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
            _tensor('F', TensorProto.FLOAT, (10, 3), INIT_F_0),
            _tensor('G1', TensorProto.FLOAT, (2, 3), INIT_G1_1),
            _tensor('T', TensorProto.FLOAT, (10, 3), INIT_T_2),
            _tensor('C3', TensorProto.FLOAT, (2, 3), INIT_C3_3),
            _tensor('A', TensorProto.FLOAT, (30, 5), INIT_A_4),
            _tensor('BQL', TensorProto.FLOAT, (3, 5), INIT_BQL_5),
            _tensor('RV', TensorProto.FLOAT, (2, 5), INIT_RV_6),
            _tensor('RM', TensorProto.FLOAT, (2, 5), INIT_RM_7),
            _tensor('R3', TensorProto.FLOAT, (3, 5), INIT_R3_8),
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
