from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task099'
TASK_NUM = 99
KAGGLE = {'score': 19.779644, 'date': '2026-07-14'}
MEMORY_BYTES = 0
PARAMS = 185
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'neurogolf_task099_projective_shared_z_from_ABw_v11'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task099_m0_p186_projective_shared_z_from_ABw_v11'
OPSETS = [('', 22)]


# U: FLOAT[30, 2], 60 value(s)
INIT_U_0 = [0.3513355553150177, 29.879791259765625, 0.6306153535842896, 30.57712745666504, 0.7287702560424805, 32.2387809753418,
 0.7475754022598267, 32.93207550048828, 0.5095817446708679, 16.527591705322266, 0.7204867005348206, 14.512864112854004,
 1.0180517435073853, 9.47440242767334, 1.1589150428771973, 11.394161224365234, 0.9670137763023376, 9.402308464050293,
 0.5570703148841858, -8.32833194732666, -1.5809160470962524, -15.49863052368164, -1.051746129989624, -50.78288650512695,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ABw: FLOAT[5, 3], 15 value(s)
INIT_ABW_1 = [2.493762731552124, -3.5286173820495605, -2.362748622894287, 0.09192677587270737, -24.289478302001953,
 1.851272463798523, 0.6562944054603577, -12.14751148223877, -0.2079406976699829, -0.9953557252883911, 8.836663246154785,
 0.8785685300827026, 2.505067825317383, 10.367061614990234, -2.1388602256774902]

# Mw: FLOAT[5, 3], 15 value(s)
INIT_MW_2 = [0.18776483833789825, 105.466064453125, -4.123520374298096, 0.74604731798172, 14.099223136901855, 0.794225811958313,
 13.90822982788086, 25.719486236572266, -4.485004901885986, 1.4522656202316284, -16.381027221679688, 1.3014458417892456,
 -1.6137195825576782, -11.474651336669922, 1.2507327795028687]

# L2: FLOAT[5, 2], 10 value(s)
INIT_L2_3 = [88.54035186767578, -0.2200903445482254, 0.143975630402565, 6.977002340136096e-05, -0.233382448554039,
 -5.647562647936866e-05, 38.61561584472656, -0.3046102523803711, -0.05760710686445236, -9.842298459261656e-05]

# RHw: FLOAT[5, 3], 15 value(s)
INIT_RHW_4 = [0.005400298163294792, -0.3421146273612976, 0.007897939532995224, -8.561314582824707, 111.92789459228516,
 35.05912780761719, -22.122486114501953, -77.53120422363281, 26.215612411499023, -0.07117889076471329,
 0.1851729303598404, 0.04518517479300499, 13.228972434997559, 196.90692138671875, 1.178916335105896]

# RWw: FLOAT[5, 3], 15 value(s)
INIT_RWW_5 = [3.042464017868042, 4.122384548187256, -2.1287131309509277, 0.8137247562408447, -19.506032943725586, 4.601645469665527,
 -0.027433577924966812, -45.27589416503906, 2.498769521713257, 1.9557744264602661, 15.596233367919922,
 -2.8637542724609375, 0.20280614495277405, 17.875656127929688, -0.17339643836021423]

# sample_shared: FLOAT[10, 5], 50 value(s)
INIT_SAMPLE_SHARED_6 = [2.88319993019104, -8.56266196933575e-05, -0.00013624291750602424, 5.933096885681152, -0.00016095662431325763,
 2.5162322521209717, 0.00012549090024549514, -0.00024432348436675966, -5.017263889312744, 0.0005034758360125124,
 8.162151336669922, 0.31751927733421326, -1.073624849319458, -4.044793605804443, -1.7669446468353271, 8.184606552124023,
 0.792633056640625, 1.1585177183151245, -4.05780553817749, -0.5683339834213257, 8.168896675109863, -0.2693731188774109,
 1.3052575588226318, -4.03735876083374, -1.6644747257232666, 8.16749382019043, 0.31068164110183716, 0.9682711362838745,
 -4.050142288208008, 1.8278210163116455, 8.143996238708496, -0.3648532032966614, -1.2450789213180542,
 -4.037321090698242, 1.6039901971817017, 8.18118953704834, 0.7372488379478455, -1.2841829061508179, -4.057799339294434,
 0.6755295991897583, 8.138299942016602, -0.7509933114051819, 1.1691855192184448, -4.03105354309082, 0.7622472643852234,
 8.140009880065918, -0.7736334204673767, -1.0295453071594238, -4.0317583084106445, -0.8337177634239197]

# P5: FLOAT[5], 5 value(s)
INIT_P5_7 = [-0.0003278824151493609, 0.7126194834709167, 0.09082413464784622, -0.004921736195683479, -1.4956718683242798]


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
        # 0: Einsum inputs=64 outputs=1
        _node(
            'Einsum',
            ['U', 'U', 'input', 'sample_shared', 'L2', 'U', 'U', 'L2', 'RHw', 'L2', 'RHw', 'Mw', 'L2', 'U', 'U',
             'U', 'L2', 'RHw', 'ABw', 'RHw', 'L2', 'U', 'U', 'input', 'sample_shared', 'U', 'U', 'L2', 'RHw',
             'ABw', 'RHw', 'L2', 'L2', 'P5', 'Mw', 'P5', 'RWw', 'Mw', 'ABw', 'RHw', 'RWw', 'RHw', 'L2', 'RHw',
             'L2', 'RHw', 'L2', 'RHw', 'L2', 'L2', 'P5', 'Mw', 'P5', 'RWw', 'Mw', 'ABw', 'sample_shared',
             'sample_shared', 'U', 'U', 'sample_shared', 'input', 'U', 'U'],
            ['output'],
            name='task099_v11_projective_shared_z_from_ABw',
            domain='',
            equation='yM,yM,nixy,ip,pE,xG,xF,vF,vL,HG,HL,qL,qM,fE,fE,fz,mD,mK,qK,gK,gC,bD,bC,nkab,kh,aB,aA,cA,cJ,qJ,eJ,eB,jz,j,jS,V,VS,Vd,hd,qN,qO,ZO,ZT,XO,XR,IN,IP,UN,UQ,wz,w,wY,W,WY,Wt,ut,oh,ou,rP,rQ,lu,nlrs,sR,sT->nors',
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
            _tensor('U', TensorProto.FLOAT, (30, 2), INIT_U_0),
            _tensor('ABw', TensorProto.FLOAT, (5, 3), INIT_ABW_1),
            _tensor('Mw', TensorProto.FLOAT, (5, 3), INIT_MW_2),
            _tensor('L2', TensorProto.FLOAT, (5, 2), INIT_L2_3),
            _tensor('RHw', TensorProto.FLOAT, (5, 3), INIT_RHW_4),
            _tensor('RWw', TensorProto.FLOAT, (5, 3), INIT_RWW_5),
            _tensor('sample_shared', TensorProto.FLOAT, (10, 5), INIT_SAMPLE_SHARED_6),
            _tensor('P5', TensorProto.FLOAT, (5,), INIT_P5_7),
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
