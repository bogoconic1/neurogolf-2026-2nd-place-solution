from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task368'
TASK_NUM = 368
KAGGLE = {'score': 19.250607, 'date': '2026-07-14'}
MEMORY_BYTES = 0
PARAMS = 314
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 't4slices'
OPSETS = [('', 18)]


# M3: FLOAT[30, 3], 90 value(s)
INIT_M3_0 = [1.0452324151992798, 0.25266769528388977, 0.03761497512459755, -0.04055504500865936, 1.03606379032135,
 0.176413431763649, 0.13063740730285645, -0.009013064205646515, 1.0565252304077148, 1.096920371055603,
 0.08442370593547821, -0.14984995126724243, 0.027961742132902145, 1.0987218618392944, 0.18380068242549896,
 0.31571704149246216, -0.07700657099485397, 1.0535207986831665, 1.0829098224639893, 0.25462785363197327,
 -0.023006660863757133, -0.011260159313678741, 1.1129100322723389, 0.1948813796043396, -0.009883700869977474,
 -0.09548895061016083, 1.1334835290908813, 1.1588325500488281, 0.08115744590759277, -0.18626464903354645, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# M4: FLOAT[30, 4], 120 value(s)
INIT_M4_1 = [1.0714820623397827, -0.04158546030521393, 0.03557227924466133, 0.19795288145542145, -0.0037360298447310925,
 1.063690185546875, 0.016971338540315628, 0.016244757920503616, -0.06666666269302368, 0.025807829573750496,
 1.1165199279785156, 0.09855566173791885, 0.14409968256950378, -0.01754012703895569, -0.12329668551683426,
 1.089206576347351, 1.1210945844650269, 0.29618531465530396, -0.052612703293561935, 0.02575431577861309,
 0.17328935861587524, 1.0371198654174805, 0.037991248071193695, 0.2849290668964386, -0.015741463750600815,
 -0.005566547624766827, 1.1123645305633545, -0.07664812356233597, 0.0011729071848094463, 0.029326023533940315,
 0.02577781118452549, 1.129637360572815, 1.1308298110961914, 0.18197517096996307, 0.019381139427423477,
 -0.07979146391153336, 0.008280517533421516, 1.1694104671478271, 0.16707199811935425, 0.03794700279831886, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# F: FLOAT[2, 10], 20 value(s)
INIT_F_2 = [1.7039376497268677, -0.012333517894148827, -0.013040645979344845, -0.0174825731664896, -0.013480146415531635,
 -0.0042046308517456055, -0.015882832929491997, -0.009869851171970367, -0.012571115046739578, -0.016314079985022545,
 -0.11608956009149551, 1.106213927268982, 0.9875046014785767, 1.0319628715515137, 1.0223309993743896,
 -1.0738197565078735, 1.0079121589660645, 0.9906474947929382, 0.9795865416526794, 0.9864920377731323]

# R3N: FLOAT[2, 2, 3, 3], 36 value(s)
INIT_R3N_3 = [1.773896336555481, -0.1483323574066162, 0.25189751386642456, 0.4202544689178467, 1.6973196268081665, -0.33979532122612,
 -0.1601094901561737, 0.17440007627010345, 1.6730707883834839, -0.5737511515617371, -0.09039905667304993,
 -0.12310466170310974, -0.060995496809482574, -0.516147792339325, 0.0040033371187746525, -0.05798615515232086,
 -0.11600768566131592, -0.5747524499893188, 0.04285138100385666, -0.10633926838636398, -0.5978482961654663,
 -0.6782439351081848, 0.1613413244485855, -0.0324852280318737, -0.0715038850903511, -0.515651524066925,
 0.07561108469963074, 0.016575107350945473, 0.2150392085313797, -0.031976472586393356, -0.04741829261183739,
 0.0157625675201416, 0.22631943225860596, 0.2332741618156433, -0.04944714903831482, -0.0019018094753846526]

# T4: FLOAT[2, 4, 4], 32 value(s)
INIT_T4_4 = [0.838128387928009, 0.8329846858978271, -0.07397191971540451, 0.05320584028959274, 0.019120026379823685,
 0.8342869877815247, 0.7424586415290833, 0.087457574903965, -0.06155845522880554, -0.02696266956627369,
 0.8115765452384949, 0.7701444625854492, 0.7794656753540039, -0.07437776029109955, -0.05606719106435776,
 0.7041741013526917, 0.18591848015785217, -0.3766207993030548, -2.3057732582092285, -2.7080471515655518,
 -2.753822088241577, -0.05378000810742378, -0.3262222707271576, -2.5611984729766846, -2.5418152809143066,
 -2.700732946395874, -0.06604763120412827, -0.41321203112602234, -0.21540246903896332, -2.2629594802856445,
 -2.793541193008423, -0.6596311926841736]

# Q4: FLOAT[4, 4], 16 value(s)
INIT_Q4_5 = [0.8969753384590149, 2.6341826915740967, -1.1570504903793335, -0.04029795527458191, -0.029147332534193993,
 1.0629627704620361, 2.6201655864715576, -1.1693450212478638, -0.47348707914352417, 0.021720178425312042,
 0.9856505990028381, 2.550837516784668, 2.3280889987945557, -0.6299399733543396, 0.20477896928787231,
 1.1264169216156006]


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
        # 0: Einsum inputs=62 outputs=1
        _node(
            'Einsum',
            ['M4', 'M3', 'R3N', 'T4', 'Q4', 'F', 'input', 'M3', 'M4', 'M4', 'M3', 'input', 'F', 'M3', 'M4',
             'R3N', 'T4', 'Q4', 'T4', 'T4', 'R3N', 'M3', 'M4', 'input', 'M3', 'M4', 'R3N', 'F', 'T4', 'T4',
             'T4', 'Q4', 'R3N', 'M4', 'M3', 'F', 'input', 'R3N', 'T4', 'T4', 'M4', 'M3', 'input', 'F', 'M3',
             'M4', 'input', 'F', 'M3', 'M4', 'T4', 'Q4', 'R3N', 'R3N', 'T4', 'T4', 'M4', 'M3', 'M4', 'M3', 'F',
             'input'],
            ['output'],
            name='output',
            domain='',
            equation='iD,iC,rrCA,rBD,DB,sz,...zic,cE,cF,aB,aA,...faj,sf,jG,jH,bbGE,bFH,HF,RLF,nFL,RnKE,yK,yL,...oxy,xI,xJ,PqIA,so,PJB,qBJ,gNQ,QN,ggOM,kQ,kO,sm,...mkv,PqWM,PXN,qNX,uN,uM,...puv,sp,vS,vT,...dul,sd,lU,lV,eTV,VT,eeUS,RnYS,RZT,nTZ,wZ,wY,hX,hW,st,...thw->...ohw',
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
            _tensor('M3', TensorProto.FLOAT, (30, 3), INIT_M3_0),
            _tensor('M4', TensorProto.FLOAT, (30, 4), INIT_M4_1),
            _tensor('F', TensorProto.FLOAT, (2, 10), INIT_F_2),
            _tensor('R3N', TensorProto.FLOAT, (2, 2, 3, 3), INIT_R3N_3),
            _tensor('T4', TensorProto.FLOAT, (2, 4, 4), INIT_T4_4),
            _tensor('Q4', TensorProto.FLOAT, (4, 4), INIT_Q4_5),
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
