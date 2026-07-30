from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task232'
TASK_NUM = 232
KAGGLE = {'score': 20.435652, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 96
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task232_transformed_cubic_moment_p96'
OPSETS = [('', 18)]


# E: FLOAT[2, 10], 20 value(s)
INIT_E_0 = [-1.1046059131622314, 0.25377655029296875, 1.2867375612258911, 0.88428795337677, 0.46243423223495483, 2.391134023666382,
 1.4371410608291626, 1.0956908464431763, 0.6734792590141296, 0.04070223495364189, 0.4399019181728363, 2.10524582862854,
 1.7298153638839722, 1.8939135074615479, 2.0384910106658936, -0.8880226612091064, 1.5815927982330322, 1.822662591934204,
 1.9677237272262573, 2.1234130859375]

# U: FLOAT[2, 2, 2, 2], 16 value(s)
INIT_U_1 = [-0.3432273268699646, -1.832964301109314, -1.7573035955429077, 1.1412556171417236, -0.20054082572460175,
 0.09939821809530258, -0.47132688760757446, -0.4158487021923065, 0.3798179030418396, 0.23758693039417267,
 -0.2470279484987259, 0.18991775810718536, 0.3046041429042816, -1.025599479675293, 0.40459179878234863,
 -0.23510661721229553]

# F: FLOAT[2, 30], 60 value(s)
INIT_F_2 = [-0.5667082667350769, -0.5962206125259399, -0.5728659629821777, -0.5851221680641174, -0.5755019783973694,
 -0.5630255937576294, -0.5347558856010437, -0.5089964270591736, -0.216640904545784, -0.21192774176597595,
 -0.2031526267528534, -0.19333979487419128, -0.18806888163089752, -0.17621839046478271, -1.2503725290298462,
 1.3790379762649536, -1.4336965084075928, 1.452876329421997, -1.4448432922363281, -1.383292317390442, 1.517167568206787,
 -1.4274235963821411, 1.353877305984497, -1.2872859239578247, 1.2195022106170654, -1.1040796041488647, 0.0, 0.0, 0.0,
 0.0, -0.31764718890190125, 0.3654615879058838, -0.3636835217475891, 0.38857921957969666, -0.4003351926803589,
 0.3901260495185852, -0.38431400060653687, 0.37679246068000793, -0.16102376580238342, 0.15604445338249207,
 -0.1515425443649292, 0.1427813470363617, -0.14083577692508698, 0.13079547882080078, -2.216139554977417,
 2.2776169776916504, -2.0333540439605713, 1.581120252609253, -0.9884382486343384, -0.322775661945343,
 -0.35401391983032227, 0.9765211343765259, -1.473382592201233, 1.825705885887146, -2.0141279697418213,
 1.9568524360656738, 0.0, 0.0, 0.0, 0.0]


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
        # 0: Einsum inputs=47 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'U', 'U', 'E', 'E', 'E', 'F', 'F',
             'input', 'E', 'E', 'E', 'F', 'F', 'input'],
            ['output'],
            name='',
            domain='',
            equation='ax,ax,ax,ax,bx,bx,bx,bx,qx,qx,px,py,qy,qy,iy,iy,iy,iy,jy,jy,jy,jy,pv,qv,qv,kv,kv,kv,kv,lv,lv,lv,lv,pqAB,pqCD,Ad,Bd,Bd,is,js,ndhs,Co,Do,Do,kw,lw,nchw->nohw',
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
            _tensor('E', TensorProto.FLOAT, (2, 10), INIT_E_0),
            _tensor('U', TensorProto.FLOAT, (2, 2, 2, 2), INIT_U_1),
            _tensor('F', TensorProto.FLOAT, (2, 30), INIT_F_2),
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
