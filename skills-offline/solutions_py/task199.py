from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task199'
TASK_NUM = 199
KAGGLE = {'score': 19.924826, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 160
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task199_m0_p165_shared_CLd_homotopy'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task199_m0_p165_shared_CLd_homotopy'
OPSETS = [('', 18)]


# X: FLOAT[3, 30], 90 value(s)
INIT_X_0 = [0.6305007338523865, 0.7958288788795471, 0.6672806143760681, 0.7506607174873352, 0.7315243482589722, 0.7582231760025024,
 0.8560236096382141, 0.7813194394111633, 0.7778112888336182, 0.8557844161987305, 0.8656007647514343, 0.877463698387146,
 0.97831130027771, 1.0291517972946167, 1.053006649017334, -0.6226453185081482, -1.0477672815322876, -1.0055385828018188,
 -0.6330715417861938, -0.9993926882743835, -0.6480165123939514, -1.0252113342285156, -1.004088044166565,
 -1.0107964277267456, 0.3554396331310272, -0.9783199429512024, -1.0365475416183472, -0.6439383625984192,
 -0.6144269704818726, -1.001645803451538, -0.10498669743537903, 1.2502057552337646, 2.070565700531006,
 3.315103530883789, 4.113044738769531, 5.06727933883667, 6.583982944488525, 6.7187604904174805, 7.389499664306641,
 8.810873031616211, 9.598989486694336, 10.386417388916016, 12.2040433883667, 13.552389144897461, 14.63344955444336,
 -7.874327182769775, -6.144005298614502, -6.254042148590088, -7.853754043579102, -6.1306657791137695, -7.7886643409729,
 -6.154562950134277, -6.269106388092041, -6.185143947601318, -10.145459175109863, -6.245111465454102,
 -6.1797027587890625, -7.829568386077881, -7.852214336395264, -6.173604965209961, 0.20479561388492584,
 1.2000247240066528, 0.1048862561583519, 1.1749807596206665, 0.043089114129543304, 1.1476082801818848,
 -0.015838271006941795, 1.128224492073059, -0.01386310625821352, 1.205908179283142, 0.03728308528661728,
 1.1323809623718262, 0.03678051754832268, 1.2086926698684692, 0.11763869971036911, -1.0502572059631348,
 -0.15394826233386993, -0.15625695884227753, -1.0441043376922607, -0.9919679760932922, -1.0484291315078735,
 -0.1554061323404312, -0.15512633323669434, -0.1549328863620758, -1.1697207689285278, -0.15478818118572235,
 -0.15535393357276917, -1.0499274730682373, -1.048882007598877, -0.15355457365512848]

# CLd: FLOAT[2, 3, 3], 18 value(s)
INIT_CLD_1 = [1.2713686227798462, 0.010803291574120522, -0.903793215751648, -0.001960034715011716, -0.0047057149931788445,
 0.02772689238190651, -1.0239473581314087, 0.039184778928756714, 0.8966366052627563, -0.04042699187994003,
 0.1230776309967041, -0.016542362049221992, -0.10181744396686554, -0.0013994865585118532, 0.0006922198808752,
 -0.03921118378639221, 0.0010672558564692736, 0.021338939666748047]

# E: FLOAT[4, 10], 40 value(s)
INIT_E_2 = [4.0134477615356445, 5.10977029800415, 3.24416446685791, 4.171768665313721, -4.338292121887207, 4.903867244720459,
 5.217957973480225, 4.717601776123047, 5.164607524871826, 4.789985656738281, -3.9322052001953125, 4.446030139923096,
 2.652108907699585, 3.4181978702545166, 3.481381893157959, 4.10599946975708, 4.442136287689209, 3.8989713191986084,
 4.279197692871094, 4.103466510772705, -0.03395877406001091, 2.053596258163452, 1.7687108516693115, 1.9198634624481201,
 0.05745965614914894, 1.9902137517929077, 2.048579692840576, 1.9563944339752197, 2.3284740447998047, 2.5194687843322754,
 -0.047251779586076736, 0.6904711127281189, 2.71327805519104, 2.2551510334014893, 0.004578264895826578,
 -1.2086766958236694, -0.17476925253868103, 1.459396243095398, -2.5397143363952637, -3.5860753059387207]

# A: FLOAT[3, 4], 12 value(s)
INIT_A_3 = [-2.7050957679748535, -2.317373752593994, 0.5027433037757874, 0.7247458696365356, 0.10427297651767731,
 0.19581077992916107, -0.034072354435920715, -0.014863085933029652, -11.195202827453613, -10.242505073547363,
 -4.94570255279541, -0.5582810640335083]


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
        # 0: Einsum inputs=41 outputs=1
        _node(
            'Einsum',
            ['X', 'CLd', 'X', 'X', 'X', 'X', 'CLd', 'X', 'X', 'CLd', 'X', 'X', 'CLd', 'X', 'X', 'X', 'input',
             'E', 'E', 'A', 'X', 'X', 'A', 'CLd', 'X', 'X', 'CLd', 'X', 'X', 'CLd', 'X', 'X', 'CLd', 'CLd',
             'CLd', 'X', 'X', 'X', 'X', 'input', 'E'],
            ['output'],
            name='output',
            domain='',
            equation='BD,RAB,AC,kC,kL,IL,SIJ,JM,uE,Ruv,vF,wG,Swz,zH,ur,wr,ndrs,gd,qd,kq,ps,ls,Kq,hbc,ce,ce,hfi,ij,ij,hNO,OP,OP,hKT,hpt,hlm,vy,zy,tx,mx,nayx,qo->noyx',
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
            _tensor('X', TensorProto.FLOAT, (3, 30), INIT_X_0),
            _tensor('CLd', TensorProto.FLOAT, (2, 3, 3), INIT_CLD_1),
            _tensor('E', TensorProto.FLOAT, (4, 10), INIT_E_2),
            _tensor('A', TensorProto.FLOAT, (3, 4), INIT_A_3),
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
