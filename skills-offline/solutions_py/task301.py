from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task301'
TASK_NUM = 301
KAGGLE = {'score': 20.522663, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 88
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'expand-simplify-recompress'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task301_p88_scaled_moment'
OPSETS = [('', 12)]


# E: FLOAT[2, 30], 60 value(s)
INIT_E_0 = [1.8461307287216187, 0.6776586174964905, 0.39770767092704773, 0.31213635206222534, 0.2654760777950287,
 0.2071974128484726, 0.17633265256881714, 0.16389665007591248, 0.13779160380363464, 0.1163458451628685,
 0.09847663342952728, 1.9268481731414795, -0.7978909015655518, -1.227443814277649, 3.9764647483825684,
 -0.884055495262146, -0.970816969871521, -0.4425455927848816, -1.820881724357605, -3.577716112136841,
 -2.4349677562713623, -3.5752522945404053, -3.0873918533325195, 4.098280429840088, -0.6003314852714539,
 -0.6272789835929871, 0.3251498341560364, -0.6028907299041748, -0.5217363238334656, 0.4809844493865967,
 0.9115179777145386, 1.0547873973846436, 1.0922490358352661, 1.1511942148208618, 1.251819133758545, 1.184800386428833,
 1.1940720081329346, 1.265661597251892, 1.2273070812225342, 1.1699819564819336, 1.1120712757110596, 2.4301700592041016,
 2.147840976715088, -1.8054494857788086, 0.18024834990501404, 2.0517733097076416, 1.9565379619598389,
 -5.3545756340026855, -0.5052873492240906, -0.9779345393180847, 0.07457847893238068, 0.6950480341911316,
 -0.9987561702728271, 0.10611303150653839, -3.4344382286071777, -3.20900559425354, -3.351102352142334,
 -3.280006170272827, -3.5851638317108154, 5.492762565612793]

# Col: FLOAT[2, 10], 20 value(s)
INIT_COL_1 = [0.8780956268310547, 0.7417417764663696, 0.7121245265007019, 0.7816208600997925, 0.7540724873542786, 0.7950780391693115,
 0.6702924370765686, 0.7291148900985718, 1.0877177715301514, 0.7498680949211121, -0.18565896153450012,
 0.9139297604560852, 0.8854090571403503, 0.9263081550598145, 0.9233136773109436, 0.9529272317886353, 0.872505247592926,
 0.8886386752128601, 1.2839109897613525, 0.9330462217330933]

# T: FLOAT[2, 2, 2], 8 value(s)
INIT_T_2 = [0.00019083265215158463, -1.2359517812728882, -0.13078755140304565, 0.8760146498680115, 1.078991413116455,
 0.5094602108001709, -1.1240825653076172, -0.019255639985203743]


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
        # 0: Einsum inputs=119 outputs=1
        _node(
            'Einsum',
            ['E', 'input', 'E', 'input', 'E', 'input', 'E', 'T', 'T', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'T',
             'E', 'E', 'E', 'E', 'E', 'E', 'E', 'T', 'T', 'T', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'T', 'T', 'T',
             'E', 'E', 'E', 'E', 'E', 'E', 'E', 'Col', 'E', 'input', 'Col', 'input', 'E', 'input', 'Col', 'E',
             'E', 'E', 'T', 'T', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'T', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'T',
             'E', 'E', 'E', 'E', 'E', 'E', 'E', 'T', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'T', 'T', 'E', 'E', 'E',
             'E', 'E', 'E', 'E', 'T', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'T', 'T', 'T', 'E', 'E', 'E', 'E', 'E',
             'E', 'E', 'E', 'E', 'E', 'input'],
            ['output'],
            name='',
            domain='',
            equation='fl,...Czl,Iz,...Bly,vy,...Aul,Hu,oIF,KIF,Fa,Ia,Ia,Ka,Ka,Ka,Ka,FIK,Kb,Ib,Ib,Fb,Fb,Fb,Fb,Fxo,Fxo,Vfo,os,fs,fs,Vs,Vs,Vs,Vs,fJc,ncJ,JfV,VM,fM,fM,JM,JM,JM,JM,VP,VQ,...PhQ,tq,...qhp,tp,...khi,gk,gE,cE,cE,cde,edt,tN,dN,dN,eN,eN,eN,eN,eOv,vS,OS,OS,eS,eS,eS,eS,ddO,OT,dT,dT,dT,dT,dT,dT,tdm,mU,dU,dU,tU,tU,tU,tU,mHR,GHR,RW,HW,HW,GW,GW,GW,GW,RHG,GX,HX,HX,RX,RX,RX,RX,RDm,RDm,vOw,wY,OY,OY,vY,vY,vY,vY,Gr,Kr,wj,...Lrj->...krj',
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
            _tensor('E', TensorProto.FLOAT, (2, 30), INIT_E_0),
            _tensor('Col', TensorProto.FLOAT, (2, 10), INIT_COL_1),
            _tensor('T', TensorProto.FLOAT, (2, 2, 2), INIT_T_2),
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
