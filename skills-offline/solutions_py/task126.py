from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task126'
TASK_NUM = 126
KAGGLE = {'score': 20.617973, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task126_p88_dot35_factored_fg'
OPSETS = [('', 18)]


# Row: FLOAT[2, 30], 60 value(s)
INIT_ROW_0 = [1.0188214778900146, 1.019377589225769, 1.0192523002624512, 1.0211955308914185, 1.0137014389038086, 1.012459397315979,
 1.0183161497116089, 1.0210773944854736, 1.0129972696304321, 1.013998031616211, 0.22815269231796265,
 -11.859105110168457, -0.05342908576130867, -0.05438898503780365, -0.0528811514377594, 0.053897783160209656,
 -0.05460206791758537, -0.05461227148771286, -0.054712485522031784, 0.054085828363895416, 0.05408337339758873,
 -0.05398483946919441, -0.05228374898433685, -0.0546146035194397, -0.05379503592848778, 0.05421184003353119,
 0.05325109139084816, 0.053385764360427856, -0.053833141922950745, -0.054234039038419724, 2.154794692993164,
 2.136789321899414, 2.1206605434417725, 2.149484634399414, -0.5774857401847839, -0.628627598285675, -0.7034189105033875,
 -0.822604775428772, -1.0148155689239502, -1.4248605966567993, -1.2815098762512207, -2.30224347114563,
 -0.02741296961903572, -0.027392569929361343, -0.027393877506256104, 0.027418972924351692, -0.027346065267920494,
 -0.027345197275280952, -0.027301179245114326, 0.026812899857759476, 0.02681165561079979, -0.027393888682127,
 -0.02738865092396736, -0.027341406792402267, -0.02667315863072872, 0.0268777497112751, 0.0274009108543396,
 0.027403786778450012, -0.02739596925675869, -0.027392365038394928]

# P: FLOAT[2, 10], 20 value(s)
INIT_P_1 = [0.3390965163707733, 0.2743513584136963, 0.1049075499176979, -0.10485157370567322, -0.33909428119659424,
 -0.2743549942970276, -0.2743159532546997, -0.10466477274894714, 0.10472074896097183, 0.27431201934814453,
 -1.9484609765640926e-06, 0.19942587614059448, 0.3224963843822479, 0.32249516248703003, -7.117261446865086e-08,
 0.19934789836406708, -0.19920536875724792, -0.322501003742218, -0.322502076625824, -0.19928301870822906]


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
        # 0: Einsum inputs=104 outputs=1
        _node(
            'Einsum',
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'Row', 'Row', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'Row', 'input',
             'Row', 'input', 'Row', 'input', 'Row', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'Row', 'Row', 'Row',
             'Row', 'Row', 'Row', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['output'],
            name='',
            domain='',
            equation='id,ic,jd,jc,kd,kc,ld,lc,md,mc,od,oc,pd,pc,qd,qc,rd,rc,sd,sc,td,tc,ud,uc,vd,vc,wd,wc,zd,zc,Ad,Ac,Bd,Bc,Cd,Cc,Dd,Dc,Ed,Ec,Fd,Fc,Gd,Gc,Hd,Hc,Id,Ic,Jd,Jc,Kd,Kc,Ld,Lc,Md,Mc,Nd,Nc,Zd,Zd,Zd,Zd,Zc,Zc,Oa,Sa,SP,SP,Oe,Oe,Re,Re,Re,RP,RP,OQ,nPQx,Oy,ndyx,TV,nUVx,Ty,Tf,Tf,Wf,Wf,Wf,WU,WU,Tb,Xb,Th,Xh,Xh,Xh,XU,XU,Tg,Tg,Yg,Yg,Yg,YU,YU->ncyx',
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
            _tensor('Row', TensorProto.FLOAT, (2, 30), INIT_ROW_0),
            _tensor('P', TensorProto.FLOAT, (2, 10), INIT_P_1),
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
