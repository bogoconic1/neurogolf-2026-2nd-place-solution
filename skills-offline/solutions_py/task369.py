from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task369'
TASK_NUM = 369
KAGGLE = {'score': 20.24641, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 116
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 7
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task369_quad_tied_m0_p116'
OPSETS = [('', 12)]


# F: FLOAT[3, 30], 90 value(s)
INIT_F_0 = [0.9795371294021606, 0.9795371294021606, 0.9795371294021606, 0.9795371294021606, 0.9795371294021606, 0.9795371294021606,
 0.9795371294021606, 0.9795371294021606, 0.9795371294021606, 0.9795371294021606, -2.1341359615325928,
 1.2348830699920654, -0.909399151802063, 1.3672057390213013, 3.4520151615142822, 2.0854899883270264,
 -1.5117331743240356, -0.3042464554309845, 1.4110301733016968, -1.8028335571289062, -0.6685513854026794,
 -0.13748258352279663, -0.3444753587245941, 1.5320453643798828, -2.759586811065674, -0.7461092472076416,
 -0.9303426146507263, -0.49019598960876465, -3.011350393295288, -0.8798863291740417, 1.0, 0.8412535190582275,
 0.41541507840156555, -0.1423148363828659, -0.6548606753349304, -0.9594929814338684, -0.9594929814338684,
 -0.6548607349395752, -0.14231501519680023, 0.41541510820388794, -1.0138226747512817, -2.7295138835906982,
 -0.9652904272079468, 2.5351717472076416, -0.1790524423122406, -0.09140460938215256, -1.157184362411499,
 -1.6133973598480225, 1.4519894123077393, 1.0179013013839722, -1.125201940536499, -0.44758543372154236,
 -0.5972381234169006, -0.0550977848470211, 1.7347221374511719, 1.6197071075439453, 1.1924612522125244,
 -0.5785092711448669, -1.1562498807907104, 1.142897129058838, 0.0, 0.5406407713890076, 0.9096319675445557,
 0.9898214340209961, 0.7557496428489685, 0.28173258900642395, -0.28173255920410156, -0.7557495832443237,
 -0.9898214340209961, -0.9096319675445557, -1.5553874969482422, -2.7505438327789307, -0.6414965987205505,
 2.5810978412628174, -0.25633925199508667, 0.053461357951164246, -0.6867532134056091, 0.9569119811058044,
 0.34363406896591187, 1.3901841640472412, -0.6456186175346375, 0.7970079779624939, 0.8011391162872314,
 -0.3039593994617462, 1.7409647703170776, -1.2264902591705322, 0.9119800925254822, -0.4352101981639862,
 -0.8430345058441162, 0.9865489602088928]

# E: FLOAT[10, 2], 20 value(s)
INIT_E_1 = [-2.540175676345825, 3.6634838581085205, -3.309408664703369, 0.7028694748878479, -3.176555633544922, 3.9824180603027344,
 -1.6043819189071655, 4.119687080383301, -3.395439386367798, 4.89874792098999, -0.8105294704437256, -1.379172682762146,
 -3.375401258468628, 4.869797706604004, -3.359476089477539, 4.846778392791748, -3.376786231994629, 4.871798992156982,
 3.3717215061187744, -4.8644795417785645]

# U: FLOAT[3, 2], 6 value(s)
INIT_U_2 = [3.489450693130493, 3.1997036933898926, -2.761023759841919, 2.0653276443481445, -0.3660913407802582,
 -1.0789366960525513]


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
        # 0: Einsum inputs=102 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'input', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'input', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'E', 'U', 'input', 'F', 'U', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'E', 'F', 'E'],
            ['output'],
            name='',
            domain='',
            equation='El,fe,ie,pe,zl,zl,El,dl,ib,zb,pb,pv,zv,fv,iv,fb,qb,qv,Ah,Bh,Ch,...awb,Aw,Cw,Ew,Bw,Dw,Au,Eu,Cu,Bu,Du,...auv,Hv,Hk,Fk,Gk,Fv,Gv,Iv,Jv,Hx,Gx,Fx,Ix,Jx,Jj,Jj,dj,Oj,Oj,Ou,Nu,Ku,Lu,Mu,Ny,Oy,Ly,Ky,Lm,Km,Mm,My,...ayx,Px,Ps,Qs,Rs,Qx,Rx,Sx,Tx,Rc,Pc,Qc,Sc,Tc,Tg,Tg,dg,Yg,Yg,Yy,an,dn,...arc,Yr,dt,Xy,Xr,Uy,Vy,Wy,Vr,Ur,VZ,UZ,WZ,ot,Wr,on->...orc',
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
            _tensor('F', TensorProto.FLOAT, (3, 30), INIT_F_0),
            _tensor('E', TensorProto.FLOAT, (10, 2), INIT_E_1),
            _tensor('U', TensorProto.FLOAT, (3, 2), INIT_U_2),
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
