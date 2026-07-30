from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task230'
TASK_NUM = 230
KAGGLE = {'score': 20.030187, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 144
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task230_tied_power_python'
OPSETS = [('', 13)]


# F: FLOAT[30, 3], 90 value(s)
INIT_F_0 = [1.0, 1.0, 0.0, 1.0, 0.9135454297065735, 0.4067366421222687, 1.0, 0.6691306233406067, 0.7431448101997375, 1.0,
 0.30901700258255005, 0.9510565400123596, 1.0, -0.10452846437692642, 0.9945219159126282, 1.0, -0.5, 0.8660253882408142,
 1.0, -0.80901700258255, 0.5877852439880371, 1.0, -0.9781476259231567, 0.2079116851091385, 1.0, -0.9781476259231567,
 -0.2079116851091385, 1.0, -0.80901700258255, -0.5877852439880371, 1.0, -0.5, -0.8660253882408142, 1.0,
 -0.10452846437692642, -0.9945219159126282, 1.0, 0.30901700258255005, -0.9510565400123596, 1.0, 0.6691306233406067,
 -0.7431448101997375, 1.0, 0.9135454297065735, -0.4067366421222687, -1.454075574874878, 1.3219903707504272,
 1.3224645853042603, 2.433337926864624, -2.256117582321167, -2.2555489540100098, 0.07332336157560349,
 2.0642974376678467, -2.122443437576294, -0.34494152665138245, 1.8458718061447144, 1.8453413248062134,
 -1.4539928436279297, 1.3221498727798462, 1.322621464729309, -1.688341498374939, 0.9432581663131714, 0.9430040121078491,
 -1.6583619117736816, -1.2418543100357056, -1.2411048412322998, -1.4542900323867798, 1.3215796947479248,
 1.3220548629760742, -1.6887648105621338, 0.9430893063545227, 0.9428550601005554, -0.3185091018676758,
 -1.9141584634780884, -1.914129376411438, -0.35501593351364136, 1.84169602394104, 1.8411483764648438,
 -0.2903102934360504, -1.9218857288360596, -1.9218372106552124, -1.6880857944488525, 0.9437226057052612,
 0.9430081844329834, -0.34796690940856934, 1.844606637954712, 1.8440710306167603, 0.06501124799251556,
 -2.1213464736938477, 2.065394401550293]

# B: FLOAT[10, 3], 30 value(s)
INIT_B_1 = [-0.47921261191368103, -1.1949130296707153, -2.3547911643981934, -24.915678024291992, 20.96159553527832,
 -73.74706268310547, -30.627958297729492, 15.913689613342285, -51.897220611572266, -11.928705215454102,
 2.8161563873291016, -7.379931926727295, -41.85441970825195, -7.962395191192627, 40.400753021240234, -69.7617416381836,
 -18.675989151000977, 86.42121887207031, 358.4921875, -115.19451141357422, 360.4654235839844, -191.8044891357422,
 82.19108581542969, -275.45257568359375, -227.22584533691406, 99.323974609375, -338.1683044433594, 241.2073516845703,
 -81.74320220947266, 258.1479797363281]

# Umat: FLOAT[3, 3], 9 value(s)
INIT_UMAT_2 = [2.711944818496704, 0.12834414839744568, -0.04009102284908295, -3.4875006675720215, -0.8483035564422607,
 -0.760603129863739, 1.775555968284607, -0.2800404131412506, -0.19930578768253326]

# Vnew: FLOAT[2, 3], 6 value(s)
INIT_VNEW_3 = [0.5, 0.18743173778057098, -0.18743173778057098, 0.5000768899917603, -6.207108020782471, 6.207108020782471]

# Wv: FLOAT[2, 3], 6 value(s)
INIT_WV_4 = [0.0871700793504715, 1.070625901222229, 0.9245001673698425, 0.017358381301164627, -0.07062584161758423,
 0.07549985498189926]

# vc: FLOAT[3], 3 value(s)
INIT_VC_5 = [-2.352030038833618, -0.42525872588157654, -0.4639684855937958]


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
        # 0: Einsum inputs=184 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'Vnew', 'F', 'F', 'F', 'Umat',
             'F', 'F', 'F', 'Wv', 'F', 'F', 'Wv', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew',
             'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Wv', 'vc', 'vc', 'vc', 'vc', 'vc', 'vc', 'vc', 'vc', 'vc',
             'vc', 'vc', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat',
             'vc', 'F', 'F', 'B', 'F', 'F', 'F', 'F', 'Wv', 'B', 'input', 'B', 'input', 'B', 'input', 'F', 'B',
             'F', 'F', 'F', 'F', 'F', 'Umat', 'F', 'F', 'Wv', 'F', 'F', 'Wv', 'Vnew', 'Vnew', 'Vnew', 'Vnew',
             'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Wv', 'vc', 'vc', 'vc', 'vc', 'vc',
             'vc', 'vc', 'vc', 'vc', 'vc', 'vc', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat',
             'Umat', 'Umat', 'Umat', 'vc', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'Vnew', 'F', 'Wv',
             'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew', 'Vnew',
             'Wv', 'vc', 'vc', 'vc', 'vc', 'vc', 'vc', 'vc', 'vc', 'vc', 'vc', 'vc', 'Umat', 'Umat', 'Umat',
             'Umat', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat', 'Umat', 'vc', 'F', 'F', 'F', 'F', 'F', 'F',
             'Umat', 'B', 'F', 'B'],
            ['output'],
            name='task230_tied_power_builder',
            domain='',
            equation='rD,fD,fD,fD,fD,fD,rF,TF,TF,rH,vH,vH,vQ,aQ,iQ,iF,iD,YA,rA,iA,rB,pB,iB,rC,qC,qn,qn,qn,qn,qn,qn,qn,qn,qn,qn,qn,qn,qn,n,n,n,n,n,n,n,n,n,n,n,nn,nn,nn,nn,nn,nn,nn,nn,nn,nn,nn,C,iC,rE,mE,iE,rG,uG,iG,az,ez,...ers,dz,...dis,cz,...crj,sM,lM,jM,sO,yO,jO,sI,ZI,jI,sJ,wJ,jJ,sK,xK,xV,xV,xV,xV,xV,xV,xV,xV,xV,xV,xV,xV,xV,V,V,V,V,V,V,V,V,V,V,V,VV,VV,VV,VV,VV,VV,VV,VV,VV,VV,VV,K,jK,sL,sP,sN,UN,UN,jN,tP,tP,tR,bR,jR,bz,bW,bW,bW,bW,bW,bW,bW,bW,bW,bW,bW,bW,bW,W,W,W,W,W,W,W,W,W,W,W,WW,WW,WW,WW,WW,WW,WW,WW,WW,WW,WW,z,kL,kL,kL,kL,kL,jL,gz,og,Sh,oh->...oij',
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
            _tensor('F', TensorProto.FLOAT, (30, 3), INIT_F_0),
            _tensor('B', TensorProto.FLOAT, (10, 3), INIT_B_1),
            _tensor('Umat', TensorProto.FLOAT, (3, 3), INIT_UMAT_2),
            _tensor('Vnew', TensorProto.FLOAT, (2, 3), INIT_VNEW_3),
            _tensor('Wv', TensorProto.FLOAT, (2, 3), INIT_WV_4),
            _tensor('vc', TensorProto.FLOAT, (3,), INIT_VC_5),
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
